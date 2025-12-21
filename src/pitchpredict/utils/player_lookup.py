"""
Utilities for looking up player names from MLBAM IDs and decoding session metadata.
"""

import json
import os
from datetime import date

import numpy as np
from tqdm import tqdm

from pitchpredict.backend.algs.deep.dataset import SESSION_START_TOKEN
from pitchpredict.backend.algs.deep.nn import TOKEN_DTYPE


def lookup_player_name(mlbam_id: int) -> str:
    """
    Look up player name from MLBAM ID using pybaseball.

    Args:
        mlbam_id: The MLBAM player ID

    Returns:
        Player name as "First Last" or "Unknown (id)" if lookup fails
    """
    import pybaseball

    try:
        df = pybaseball.playerid_reverse_lookup([mlbam_id], key_type="mlbam")
        if not df.empty:
            row = df.iloc[0]
            first = str(row['name_first']).title()
            last = str(row['name_last']).title()
            return f"{first} {last}"
    except Exception as e:
        print(f"Warning: Could not look up ID {mlbam_id}: {e}")

    return f"Unknown ({mlbam_id})"


def decode_game_date(normalized_value: float) -> str:
    """
    Decode normalized game_date back to YYYY-MM-DD.

    The encoding maps dates from 2015-01-01 to 2025-11-18 into [0, 1].

    Args:
        normalized_value: Float in [0, 1] representing the date

    Returns:
        ISO format date string (YYYY-MM-DD)
    """
    min_date = date.fromisoformat("2015-01-01").toordinal()
    max_date = date.fromisoformat("2025-11-18").toordinal()
    val = max(0.0, min(1.0, normalized_value))
    return date.fromordinal(int(val * (max_date - min_date) + min_date)).isoformat()


class PlayerNameCache:
    """
    Cache MLBAM ID -> player name mappings to a JSON file.

    This avoids repeated API calls to pybaseball for the same player IDs.
    """

    def __init__(self, cache_path: str = "player_names.json"):
        self.cache_path = cache_path
        self.cache: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from file if it exists."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                # JSON keys are strings, convert to int
                self.cache = {int(k): v for k, v in json.load(f).items()}

    def _save(self) -> None:
        """Save cache to file."""
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def get(self, mlbam_id: int) -> str | None:
        """Get a cached name, or None if not cached."""
        return self.cache.get(mlbam_id)

    def set(self, mlbam_id: int, name: str) -> None:
        """Set a name in the cache and save."""
        self.cache[mlbam_id] = name
        self._save()

    def batch_lookup(self, mlbam_ids: list[int]) -> dict[int, str]:
        """
        Look up multiple IDs, fetching missing ones from pybaseball.

        Args:
            mlbam_ids: List of MLBAM player IDs

        Returns:
            Dict mapping ID -> player name
        """
        results = {}
        missing = []

        for mid in mlbam_ids:
            if mid in self.cache:
                results[mid] = self.cache[mid]
            else:
                missing.append(mid)

        if missing:
            print(f"Looking up {len(missing)} new player names...")
            for mid in tqdm(missing, desc="Player lookups"):
                name = lookup_player_name(mid)
                self.cache[mid] = name
                results[mid] = name
            self._save()
            print(f"Cache now has {len(self.cache)} players")

        return results


def load_session_info(data_dir: str) -> list[dict]:
    """
    Load pitcher_id, batter_ids, game_date for each session.

    Args:
        data_dir: Path to data directory containing pitch_seq.bin and context files

    Returns:
        List of dicts with pitcher_id, batter_ids (list of unique batters in order seen), and game_date
    """
    from pitchpredict.backend.algs.deep.dataset import SESSION_END_TOKEN

    # Load tokens to find session boundaries
    tokens_path = os.path.join(data_dir, "pitch_seq.bin")
    tokens = np.memmap(tokens_path, dtype=TOKEN_DTYPE, mode="r")
    session_starts = np.where(tokens == SESSION_START_TOKEN)[0]
    session_ends = np.where(tokens == SESSION_END_TOKEN)[0]

    # Load context fields
    pitcher_ids = np.memmap(
        os.path.join(data_dir, "pitch_context_pitcher_id.bin"),
        dtype=np.int32,
        mode="r",
    )
    batter_ids = np.memmap(
        os.path.join(data_dir, "pitch_context_batter_id.bin"),
        dtype=np.int32,
        mode="r",
    )
    game_dates = np.memmap(
        os.path.join(data_dir, "pitch_context_game_date.bin"),
        dtype=np.float32,
        mode="r",
    )

    sessions = []
    for i, start in enumerate(session_starts):
        # Determine session end
        if i < len(session_ends):
            end = int(session_ends[i])
        else:
            end = len(tokens)

        # Get unique batters in order seen
        session_batter_ids = batter_ids[start:end]
        unique_batters = []
        seen = set()
        for bid in session_batter_ids:
            bid = int(bid)
            if bid not in seen:
                seen.add(bid)
                unique_batters.append(bid)

        sessions.append({
            "pitcher_id": int(pitcher_ids[start]),
            "batter_ids": unique_batters,
            "batter_id": unique_batters[0] if unique_batters else 0,  # First batter for backwards compat
            "game_date": decode_game_date(float(game_dates[start])),
        })

    return sessions


if __name__ == "__main__":
    # Test examples
    print("Testing player_lookup utilities\n")

    # Test player lookup with cache
    cache = PlayerNameCache("test_player_cache.json")

    # Test with some example MLBAM IDs (these are real player IDs)
    test_ids = [543037, 519242, 605400]  # Kershaw, Trout, Ohtani
    print("Looking up test player IDs...")
    names = cache.batch_lookup(test_ids)
    print("\nResults:")
    for mid, name in names.items():
        print(f"  {mid} -> {name}")

    # Test session info loading
    test_dir = "/raid/kline/pitchpredict/.pitchpredict_data_fixed/test"
    if os.path.exists(test_dir):
        print(f"\nLoading session info from {test_dir}...")
        sessions = load_session_info(test_dir)
        print(f"Loaded {len(sessions)} sessions")
        print("\nFirst 5 sessions:")
        for i, s in enumerate(sessions[:5]):
            print(f"  [{i}] Pitcher {s['pitcher_id']} vs Batter {s['batter_id']} on {s['game_date']}")
    else:
        print(f"\nSkipping session info test - {test_dir} not found")

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

"""Tests for CLI argument parsing and command dispatch."""

import argparse

import pytest

from pitchpredict.cli import build_parser


class TestCLIParser:
    """Test CLI argument parsing."""

    @pytest.fixture
    def parser(self) -> argparse.ArgumentParser:
        return build_parser()

    def test_parser_builds(self, parser: argparse.ArgumentParser):
        """Test that the parser builds without errors."""
        assert parser is not None
        assert parser.prog == "pitchpredict"

    def test_serve_command(self, parser: argparse.ArgumentParser):
        """Test serve command parsing."""
        args = parser.parse_args(["serve"])
        assert args.command == "serve"
        assert args.host == "0.0.0.0"
        assert args.port == 8056
        assert args.reload is False

    def test_serve_command_with_args(self, parser: argparse.ArgumentParser):
        """Test serve command with custom arguments."""
        args = parser.parse_args(
            ["serve", "--host", "127.0.0.1", "--port", "9000", "--reload"]
        )
        assert args.command == "serve"
        assert args.host == "127.0.0.1"
        assert args.port == 9000
        assert args.reload is True

    def test_serve_command_short_args(self, parser: argparse.ArgumentParser):
        """Test serve command with short argument forms."""
        args = parser.parse_args(["serve", "-H", "localhost", "-p", "8080", "-r"])
        assert args.host == "localhost"
        assert args.port == 8080
        assert args.reload is True

    def test_version_command(self, parser: argparse.ArgumentParser):
        """Test version command parsing."""
        args = parser.parse_args(["version"])
        assert args.command == "version"
        assert hasattr(args, "func")

    def test_predict_pitcher_command(self, parser: argparse.ArgumentParser):
        """Test predict pitcher command parsing."""
        args = parser.parse_args(["predict", "pitcher", "Corbin Burnes", "Aaron Judge"])
        assert args.command == "predict"
        assert args.predict_command == "pitcher"
        assert args.pitcher == "Corbin Burnes"
        assert args.batter == "Aaron Judge"
        assert args.balls == 0
        assert args.strikes == 0
        assert args.algorithm == "similarity"
        assert args.format == "rich"

    def test_predict_pitcher_with_context(self, parser: argparse.ArgumentParser):
        """Test predict pitcher command with game context."""
        args = parser.parse_args(
            [
                "predict",
                "pitcher",
                "592789",
                "660271",
                "--balls",
                "2",
                "--strikes",
                "1",
                "--outs",
                "1",
                "--inning",
                "5",
                "--date",
                "2024-06-15",
                "--algorithm",
                "similarity",
                "--format",
                "json",
                "--verbose",
            ]
        )
        assert args.pitcher == "592789"
        assert args.batter == "660271"
        assert args.balls == 2
        assert args.strikes == 1
        assert args.outs == 1
        assert args.inning == 5
        assert args.date == "2024-06-15"
        assert args.format == "json"
        assert args.verbose is True

    def test_predict_batter_command(self, parser: argparse.ArgumentParser):
        """Test predict batter command parsing."""
        args = parser.parse_args(
            [
                "predict",
                "batter",
                "Aaron Judge",
                "Corbin Burnes",
                "FF",
                "95.5",
                "0.5",
                "2.5",
            ]
        )
        assert args.command == "predict"
        assert args.predict_command == "batter"
        assert args.batter == "Aaron Judge"
        assert args.pitcher == "Corbin Burnes"
        assert args.type == "FF"
        assert args.speed == 95.5
        assert args.release_x == 0.5
        assert args.release_z == 2.5

    def test_predict_batted_ball_command(self, parser: argparse.ArgumentParser):
        """Test predict batted-ball command parsing."""
        args = parser.parse_args(
            [
                "predict",
                "batted-ball",
                "105.2",
                "25.0",
                "--spray-angle",
                "10.0",
                "--bb-type",
                "fly_ball",
            ]
        )
        assert args.command == "predict"
        assert args.predict_command == "batted-ball"
        assert args.launch_speed == 105.2
        assert args.launch_angle == 25.0
        assert args.spray_angle == 10.0
        assert args.bb_type == "fly_ball"

    def test_player_lookup_command(self, parser: argparse.ArgumentParser):
        """Test player lookup command parsing."""
        args = parser.parse_args(["player", "lookup", "ohtani"])
        assert args.command == "player"
        assert args.player_command == "lookup"
        assert args.name == "ohtani"
        assert args.exact is False
        assert args.limit == 5
        assert args.format == "rich"

    def test_player_lookup_exact(self, parser: argparse.ArgumentParser):
        """Test player lookup with exact matching."""
        args = parser.parse_args(
            [
                "player",
                "lookup",
                "Smith",
                "--exact",
                "--limit",
                "10",
                "--format",
                "json",
            ]
        )
        assert args.name == "Smith"
        assert args.exact is True
        assert args.limit == 10
        assert args.format == "json"

    def test_player_info_command(self, parser: argparse.ArgumentParser):
        """Test player info command parsing."""
        args = parser.parse_args(["player", "info", "660271"])
        assert args.command == "player"
        assert args.player_command == "info"
        assert args.mlbam_id == 660271

    def test_cache_status_command(self, parser: argparse.ArgumentParser):
        """Test cache status command parsing."""
        args = parser.parse_args(["cache", "status"])
        assert args.command == "cache"
        assert args.cache_command == "status"
        assert args.cache_dir == ".pitchpredict_cache"

    def test_cache_clear_command(self, parser: argparse.ArgumentParser):
        """Test cache clear command parsing."""
        args = parser.parse_args(
            ["cache", "clear", "--confirm", "--category", "pitcher"]
        )
        assert args.command == "cache"
        assert args.cache_command == "clear"
        assert args.confirm is True
        assert args.category == "pitcher"

    def test_cache_warm_command(self, parser: argparse.ArgumentParser):
        """Test cache warm command parsing."""
        args = parser.parse_args(
            ["cache", "warm", "Shohei Ohtani", "--type", "pitcher"]
        )
        assert args.command == "cache"
        assert args.cache_command == "warm"
        assert args.player == "Shohei Ohtani"
        assert args.type == "pitcher"

    def test_cache_warm_batted_ball_command(self, parser: argparse.ArgumentParser):
        """Test cache warm batted-ball command parsing."""
        args = parser.parse_args(
            ["cache", "warm", "--type", "batted-ball", "--seasons", "2"]
        )
        assert args.command == "cache"
        assert args.cache_command == "warm"
        assert args.player is None
        assert args.type == "batted-ball"
        assert args.seasons == 2

    def test_invalid_ball_count_rejected(self, parser: argparse.ArgumentParser):
        """Test that invalid ball counts are rejected."""
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["predict", "pitcher", "pitcher", "batter", "--balls", "4"]
            )

    def test_invalid_strike_count_rejected(self, parser: argparse.ArgumentParser):
        """Test that invalid strike counts are rejected."""
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["predict", "pitcher", "pitcher", "batter", "--strikes", "3"]
            )

    def test_missing_required_args_rejected(self, parser: argparse.ArgumentParser):
        """Test that missing required args cause an error."""
        with pytest.raises(SystemExit):
            # predict pitcher requires pitcher and batter args
            parser.parse_args(["predict", "pitcher"])

    def test_command_has_func(self, parser: argparse.ArgumentParser):
        """Test that each command has a handler function."""
        for cmd_args in [
            ["serve"],
            ["version"],
            ["predict", "pitcher", "p", "b"],
            ["predict", "batter", "b", "p", "FF", "95", "0", "2.5"],
            ["predict", "batted-ball", "100", "25"],
            ["player", "lookup", "name"],
            ["player", "info", "12345"],
            ["cache", "status"],
            ["cache", "clear"],
            ["cache", "warm", "player"],
            ["cache", "warm", "--type", "batted-ball"],
        ]:
            args = parser.parse_args(cmd_args)
            assert hasattr(args, "func"), f"Command {cmd_args} missing func attribute"
            assert callable(args.func), f"Command {cmd_args} func is not callable"

from src import config


def test_seasons_cover_advanced_stat_era():
    assert config.SEASONS[0] == "1718"
    assert "2526" in config.SEASONS
    assert len(config.SEASONS) == 9


def test_position_groups_map_fbref_codes():
    assert config.POSITION_GROUPS["FW"] == "attacker"
    assert config.POSITION_GROUPS["DF"] == "defender"
    assert config.POSITION_GROUPS["GK"] == "keeper"


def test_targets_are_ordered():
    assert config.TARGETS == ("minutes", "output", "output_share", "retention")

from vi_chess.universes.base import Universe, register, get, names

# Importing each module triggers @register and populates the registry.
import vi_chess.universes.balanced  # noqa: F401
import vi_chess.universes.material_greedy  # noqa: F401
import vi_chess.universes.aggression  # noqa: F401
import vi_chess.universes.endgame_purist  # noqa: F401
import vi_chess.universes.mobility  # noqa: F401
import vi_chess.universes.structural  # noqa: F401
import vi_chess.universes.chaos  # noqa: F401

__all__ = ["Universe", "register", "get", "names"]

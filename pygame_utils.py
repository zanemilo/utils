#!/usr/bin/env python3
"""
pygame_utils.py

Author: Zane Deso
Purpose: A convenience toolkit for Pygame projects, offering reusable
         functionality for both top-down and platformer 2D games.
         This includes image loading with caching and configurable colorkey,
         animations (with an optional on_complete callback), particle effects,
         and tilemap handling for collisions and rendering.
         Additionally, an Editor class has been integrated to allow in-engine
         level editing.

Enhancements:
- Integrated error handling using the @handle_errors decorator.
- Configurable asset paths via the 'BASE_IMG_PATH' environment variable.
- Caching of loaded images to improve performance.
- Improved handling of mutable default arguments.
- Logging integration to record key operations.
- Asynchronous asset loading via async_load_image.
- Optional error propagation in load_image via propagate_error flag.
- Detailed docstrings and usage examples.
- Integrated Editor for live editing.

Usage Example:
    import pygame
    import pygame_utils as pgutils

    # For non-editor projects, you can simply use:
    sprite_img = pgutils.load_image("player.png", colorkey=(0, 0, 0))
    # etc.

    # To launch the integrated level editor:
    if __name__ == "__main__":
        editor = pgutils.Editor()
        editor.run()

License: MIT
"""

import os
import sys
import json
import pygame
import logging
import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor
from error_handling import handle_errors, UtilsError

# ---------------------------------------------------------------------
# CONFIGURATION & GLOBALS
# ---------------------------------------------------------------------
BASE_IMG_PATH = os.getenv("BASE_IMG_PATH", "data/images/")
_image_cache = {}
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------
# ASYNCHRONOUS ASSET LOADING
# ---------------------------------------------------------------------
@handle_errors(default_return=None)
def load_image(path: str, colorkey: tuple = (0, 0, 0), propagate_error: bool = False) -> pygame.Surface:
    """
    Load a single image with caching. Optionally propagates errors.
    
    Parameters:
        path (str): Relative path to the image file.
        colorkey (tuple): RGB color to treat as transparent. Set to None to disable.
        propagate_error (bool): If True, raises a UtilsError on failure; otherwise, returns None.
    
    Returns:
        pygame.Surface: The loaded image, or None if an error occurs.
    """
    full_path = os.path.join(BASE_IMG_PATH, path)
    if full_path in _image_cache:
        logging.debug("Image cache hit for: %s", full_path)
        return _image_cache[full_path]
    try:
        img = pygame.image.load(full_path).convert()
        if colorkey is not None:
            img.set_colorkey(colorkey)
        _image_cache[full_path] = img
        logging.info("Loaded image: %s", full_path)
        return img
    except Exception as e:
        logging.exception("Failed to load image: %s", full_path)
        if propagate_error:
            raise UtilsError(f"Error loading image {full_path}") from e
        return None

async def async_load_image(path: str, colorkey: tuple = (0, 0, 0), propagate_error: bool = False) -> pygame.Surface:
    """
    Asynchronously load an image by offloading the work to a thread.
    
    Parameters:
        path (str): Relative path to the image file.
        colorkey (tuple): RGB color to treat as transparent.
        propagate_error (bool): See load_image.
    
    Returns:
        pygame.Surface: The loaded image.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, load_image, path, colorkey, propagate_error
    )
    return result

def load_images(path: str, colorkey: tuple = (0, 0, 0)) -> list:
    """
    Load multiple images from a directory with caching.
    
    Parameters:
        path (str): Directory (relative to BASE_IMG_PATH) containing images.
        colorkey (tuple): RGB color to treat as transparent.
    
    Returns:
        list[pygame.Surface]: List of loaded images.
    """
    full_path = os.path.join(BASE_IMG_PATH, path)
    images = []
    try:
        if not os.path.isdir(full_path):
            logging.error("Directory not found: %s", full_path)
            return images
        for img_name in sorted(os.listdir(full_path)):
            img_path = os.path.join(path, img_name)
            img = load_image(img_path, colorkey=colorkey)
            if img:
                images.append(img)
        logging.info("Loaded %d images from %s", len(images), full_path)
    except Exception as e:
        logging.exception("Error loading images from directory: %s", full_path)
    return images

# ---------------------------------------------------------------------
# ANIMATION
# ---------------------------------------------------------------------
class Animation:
    """
    Manages a series of images (frames) for animations.
    
    Attributes:
        images (list[pygame.Surface]): Frames of the animation.
        img_duration (int): Number of updates each frame is displayed.
        loop (bool): Whether the animation loops.
        on_complete (callable): Optional callback invoked when a non-looping animation completes.
        done (bool): Indicates completion (for non-looping animations).
        frame (int): Internal counter for tracking current frame progress.
    """
    def __init__(self, images: list, img_dur: int = 5, loop: bool = True, on_complete=None):
        self.images = images
        self.img_duration = img_dur
        self.loop = loop
        self.on_complete = on_complete
        self.done = False
        self.frame = 0
        logging.debug("Initialized Animation: %d frames, duration=%d, loop=%s",
                      len(images), img_dur, loop)

    def copy(self) -> "Animation":
        anim_copy = Animation(self.images, self.img_duration, self.loop, self.on_complete)
        anim_copy.frame = self.frame
        return anim_copy

    def update(self) -> None:
        """
        Advances the animation frame; calls on_complete callback when done.
        """
        total_frames = self.img_duration * len(self.images)
        if self.loop:
            self.frame = (self.frame + 1) % total_frames
        else:
            self.frame = min(self.frame + 1, total_frames - 1)
            if self.frame >= total_frames - 1:
                self.done = True
                if callable(self.on_complete):
                    logging.info("Animation complete; invoking on_complete callback.")
                    self.on_complete()
        logging.debug("Animation updated: frame %d/%d, done=%s", self.frame, total_frames, self.done)

    def img(self) -> pygame.Surface:
        """
        Retrieves the current frame's image.
        """
        index = int(self.frame // self.img_duration)
        return self.images[index]

# ---------------------------------------------------------------------
# PARTICLE
# ---------------------------------------------------------------------
class Particle:
    """
    Provides a basic particle for visual effects.
    
    Attributes:
        game: Reference to the main game instance (with assets and configurations).
        type (str): Key for the particle asset in game.assets.
        pos (list[float]): Particle position as [x, y].
        velocity (list[float]): Movement vector per update.
        animation (Animation): Controls particle animation frames.
    """
    def __init__(self, game, p_type: str, pos, velocity=None, frame=0):
        self.game = game
        self.type = p_type
        self.pos = list(pos)
        if velocity is None:
            velocity = [0, 0]
        self.velocity = list(velocity)
        try:
            self.animation = self.game.assets[f"particle/{p_type}"].copy()
            self.animation.frame = frame
            logging.info("Initialized Particle: type='%s', position=%s", p_type, pos)
        except Exception as e:
            logging.exception("Error initializing Particle of type '%s'", p_type)
            self.animation = Animation([])

    def update(self) -> bool:
        """
        Updates the particle's position and animation.

        Returns:
            bool: True if the particle's animation is finished and it should be removed.
        """
        if self.animation.done:
            return True
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.animation.update()
        return False

    def render(self, surf: pygame.Surface, offset=(0, 0)) -> None:
        """
        Renders the particle onto a surface.

        Parameters:
            surf (pygame.Surface): The target surface.
            offset (tuple[int, int]): Camera offset for positioning.
        """
        img = self.animation.img()
        draw_x = self.pos[0] - offset[0] - img.get_width() // 2
        draw_y = self.pos[1] - offset[1] - img.get_height() // 2
        surf.blit(img, (draw_x, draw_y))

# ---------------------------------------------------------------------
# TILEMAP & RELATED CONSTANTS
# ---------------------------------------------------------------------
PHYSICS_TILES = {"grass", "stone"}      # Tiles that are solid for collision purposes.
AUTOTILE_TYPES = {"grass", "stone"}       # Tiles eligible for auto-tiling.

# Offsets for a 3x3 grid (neighbors) around a tile.
NEIGHBOR_OFFSETS = [
    (-1, 0), (-1, -1), (0, -1),
    (1, -1), (1, 0),   (0, 0),
    (-1, 1), (0, 1),   (1, 1)
]

# Mapping for auto-tiling: neighbor configuration -> variant index.
AUTOTILE_MAP = {
    tuple(sorted([(1, 0), (0, 1)])): 0,
    tuple(sorted([(1, 0), (0, 1), (-1, 0)])): 1,
    tuple(sorted([(-1, 0), (0, 1)])): 2,
    tuple(sorted([(-1, 0), (0, -1), (0, 1)])): 3,
    tuple(sorted([(-1, 0), (0, -1)])): 4,
    tuple(sorted([(-1, 0), (0, -1), (1, 0)])): 5,
    tuple(sorted([(1, 0), (0, -1)])): 6,
    tuple(sorted([(1, 0), (0, -1), (0, 1)])): 7,
    tuple(sorted([(1, 0), (-1, 0), (0, 1), (0, -1)])): 8,
}

class Tilemap:
    """
    Manages a grid-based tilemap for both top-down and platformer games, including
    collision detection, auto-tiling, and rendering.

    Attributes:
        game: Reference to the main game instance (for asset access).
        tile_size (int): Pixel size of each tile.
        tilemap (dict): Mapping "x;y" -> tile data (type, variant, pos).
        offgrid_tiles (list): Decorative tiles not aligned to the grid.
    """
    def __init__(self, game, tile_size=16):
        self.game = game
        self.tile_size = tile_size
        self.tilemap = {}       # e.g., "3;4": { 'type': 'grass', 'variant': 0, 'pos': [3, 4] }
        self.offgrid_tiles = [] # Tiles placed off the grid

    @handle_errors(default_return=None)
    def load(self, path: str) -> None:
        """
        Loads tilemap data from a JSON file.

        Parameters:
            path (str): Path to the JSON file containing tilemap data.
        """
        try:
            with open(path, "r") as f:
                map_data = json.load(f)
            self.tilemap = map_data.get("tilemap", {})
            self.tile_size = map_data.get("tile_size", self.tile_size)
            self.offgrid_tiles = map_data.get("offgrid", [])
            logging.info("Tilemap loaded from %s", path)
        except Exception as e:
            logging.exception("Failed to load tilemap from %s", path)

    @handle_errors(default_return=False)
    def save(self, path: str) -> bool:
        """
        Saves the current tilemap to a JSON file.

        Parameters:
            path (str): Path for saving the tilemap JSON file.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        map_data = {
            "tilemap": self.tilemap,
            "tile_size": self.tile_size,
            "offgrid": self.offgrid_tiles
        }
        try:
            with open(path, "w") as f:
                json.dump(map_data, f)
            logging.info("Tilemap saved to %s", path)
            return True
        except Exception as e:
            logging.exception("Failed to save tilemap to %s", path)
            return False

    def tile_around(self, pos) -> list:
        """
        Returns a list of up to 9 tiles surrounding the given pixel position.
        Useful for collision detection.

        Parameters:
            pos (tuple[float, float]): Pixel coordinates.

        Returns:
            list: Tile dictionaries around the given position.
        """
        tiles = []
        tile_loc = (int(pos[0] // self.tile_size), int(pos[1] // self.tile_size))
        for offset in NEIGHBOR_OFFSETS:
            check_loc = f"{tile_loc[0] + offset[0]};{tile_loc[1] + offset[1]}"
            if check_loc in self.tilemap:
                tiles.append(self.tilemap[check_loc])
        return tiles

    def solid_check(self, pos) -> dict:
        """
        Checks whether the tile at the given pixel position is solid (part of PHYSICS_TILES).

        Parameters:
            pos (tuple[float, float]): Pixel coordinates.

        Returns:
            dict or None: Tile data if solid; otherwise, None.
        """
        tile_loc = f"{int(pos[0] // self.tile_size)};{int(pos[1] // self.tile_size)}"
        if tile_loc in self.tilemap:
            tile_type = self.tilemap[tile_loc].get("type", "")
            if tile_type in PHYSICS_TILES:
                return self.tilemap[tile_loc]
        return None

    def physics_rects_around(self, pos) -> list:
        """
        Creates pygame.Rect objects for all solid tiles around a given pixel position.

        Parameters:
            pos (tuple[float, float]): Pixel coordinates.

        Returns:
            list[pygame.Rect]: List of collision rectangles.
        """
        rects = []
        for tile in self.tile_around(pos):
            if tile.get("type", "") in PHYSICS_TILES:
                rects.append(pygame.Rect(tile["pos"][0] * self.tile_size,
                                         tile["pos"][1] * self.tile_size,
                                         self.tile_size, self.tile_size))
        return rects

    def autotile(self) -> None:
        """
        Applies auto-tiling logic to eligible tiles by adjusting their 'variant' field
        based on adjacent tiles of the same type.
        """
        for loc_str, tile in self.tilemap.items():
            if tile.get("type", "") in AUTOTILE_TYPES:
                neighbors = set()
                x, y = tile.get("pos", [0, 0])
                for shift in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                    check_str = f"{x + shift[0]};{y + shift[1]}"
                    if check_str in self.tilemap:
                        if self.tilemap[check_str].get("type", "") == tile["type"]:
                            neighbors.add(shift)
                neighbors_tuple = tuple(sorted(neighbors))
                if neighbors_tuple in AUTOTILE_MAP:
                    tile["variant"] = AUTOTILE_MAP[neighbors_tuple]

    def extract(self, id_pairs, keep: bool = False) -> list:
        """
        Finds (and optionally removes) tiles matching specified (type, variant) pairs.

        Parameters:
            id_pairs (list[tuple[str, int]]): List of (type, variant) pairs to match.
            keep (bool): If False, matching tiles are removed from the tilemap.

        Returns:
            list: Matching tile data dictionaries.
        """
        matches = []
        # Check offgrid tiles.
        for tile in self.offgrid_tiles.copy():
            if (tile.get("type", ""), tile.get("variant", -1)) in id_pairs:
                matches.append(tile.copy())
                if not keep:
                    self.offgrid_tiles.remove(tile)
        # Check grid-based tiles.
        for loc in list(self.tilemap.keys()):
            tile = self.tilemap[loc]
            if (tile.get("type", ""), tile.get("variant", -1)) in id_pairs:
                tile_copy = tile.copy()
                tile_copy["pos"] = [
                    tile["pos"][0] * self.tile_size,
                    tile["pos"][1] * self.tile_size
                ]
                matches.append(tile_copy)
                if not keep:
                    del self.tilemap[loc]
        return matches

    def render(self, surf: pygame.Surface, offset=(0, 0)) -> None:
        """
        Renders both off-grid and grid-based tiles onto a surface.
        Only grid tiles within the visible region are rendered.

        Parameters:
            surf (pygame.Surface): The target surface.
            offset (tuple[int, int]): Camera offset for rendering.
        """
        # Render offgrid tiles.
        for tile in self.offgrid_tiles:
            try:
                tile_img = self.game.assets[tile["type"]][tile["variant"]]
                surf.blit(tile_img, (tile["pos"][0] - offset[0], tile["pos"][1] - offset[1]))
            except Exception as e:
                logging.exception("Error rendering offgrid tile: %s", tile)
        # Render grid-based tiles.
        screen_width, screen_height = surf.get_size()
        left_tile = offset[0] // self.tile_size
        right_tile = (offset[0] + screen_width) // self.tile_size + 1
        top_tile = offset[1] // self.tile_size
        bottom_tile = (offset[1] + screen_height) // self.tile_size + 1

        for x in range(left_tile, right_tile):
            for y in range(top_tile, bottom_tile):
                loc_str = f"{x};{y}"
                if loc_str in self.tilemap:
                    tile = self.tilemap[loc_str]
                    try:
                        draw_x = tile["pos"][0] * self.tile_size - offset[0]
                        draw_y = tile["pos"][1] * self.tile_size - offset[1]
                        tile_img = self.game.assets[tile["type"]][tile["variant"]]
                        surf.blit(tile_img, (draw_x, draw_y))
                    except Exception as e:
                        logging.exception("Error rendering tile at %s", loc_str)

# ---------------------------------------------------------------------
# EDITOR CLASS INTEGRATION
# ---------------------------------------------------------------------
RENDER_SCALE = 2.0  # Determines the multiplicative size of each pixel

class Editor:
    """
    A simple level editor using the utilities provided in this module.
    This class creates a window where you can place tiles on a grid (or off-grid),
    scroll around, and save/load your map.
    """
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Editor")
        self.screen = pygame.display.set_mode((640, 480))
        self.display = pygame.Surface((320, 240))
        self.clock = pygame.time.Clock()

        # Load assets from various tile directories.
        self.assets = {
            'decor': load_images('tiles/decor'),
            'grass': load_images('tiles/grass'),
            'large_decor': load_images('tiles/large_decor'),
            'stone': load_images('tiles/stone'),
            'spawners': load_images('tiles/spawners'),
        }

        self.movement = [False, False, False, False]  # [Left, Right, UP, Down]
        self.tilemap = Tilemap(self, tile_size=16)
        try:
            self.tilemap.load('map.json')
        except FileNotFoundError:
            pass

        self.scroll = [0, 0]
        self.scroll_speed = 3

        self.tile_list = list(self.assets)
        self.tile_group = 0
        self.tile_variant = 0

        self.clicking = False
        self.right_clicking = False
        self.shift = False
        self.ongrid = True

    def run(self):
        while True:
            self.display.fill((0, 0, 0))
            # Update camera scroll based on key inputs.
            self.scroll[0] += (self.movement[1] - self.movement[0]) * self.scroll_speed
            self.scroll[1] += (self.movement[3] - self.movement[2]) * self.scroll_speed
            render_scroll = (int(self.scroll[0]), int(self.scroll[1]))

            # Render tilemap.
            self.tilemap.render(self.display, offset=render_scroll)

            # Prepare a preview of the current tile.
            current_tile_img = self.assets[self.tile_list[self.tile_group]][self.tile_variant].copy()
            current_tile_img.set_alpha(100)  # Make the preview semi-transparent.

            mpos = pygame.mouse.get_pos()
            mpos = (mpos[0] / RENDER_SCALE, mpos[1] / RENDER_SCALE)
            tile_pos = (int((mpos[0] + self.scroll[0]) // self.tilemap.tile_size),
                        int((mpos[1] + self.scroll[1]) // self.tilemap.tile_size))

            if self.ongrid:
                # Snap tile preview to grid.
                self.display.blit(current_tile_img,
                                  (tile_pos[0] * self.tilemap.tile_size - self.scroll[0],
                                   tile_pos[1] * self.tilemap.tile_size - self.scroll[1]))
            else:
                self.display.blit(current_tile_img, mpos)

            if self.clicking and self.ongrid:
                tile_key = f"{tile_pos[0]};{tile_pos[1]}"
                self.tilemap.tilemap[tile_key] = {
                    'type': self.tile_list[self.tile_group],
                    'variant': self.tile_variant,
                    'pos': tile_pos
                }
            if self.right_clicking:
                tile_key = f"{tile_pos[0]};{tile_pos[1]}"
                if tile_key in self.tilemap.tilemap:
                    del self.tilemap.tilemap[tile_key]
                # Remove offgrid tiles that collide with the mouse.
                for tile in self.tilemap.offgrid_tiles.copy():
                    tile_img = self.assets[tile['type']][tile['variant']]
                    tile_rect = pygame.Rect(tile['pos'][0] - self.scroll[0],
                                            tile['pos'][1] - self.scroll[1],
                                            tile_img.get_width(),
                                            tile_img.get_height())
                    if tile_rect.collidepoint(mpos):
                        self.tilemap.offgrid_tiles.remove(tile)

            # Draw a small preview on screen.
            self.display.blit(current_tile_img, (5, 5))
            
            # Event handling.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.clicking = True
                        if not self.ongrid:
                            self.tilemap.offgrid_tiles.append({
                                'type': self.tile_list[self.tile_group],
                                'variant': self.tile_variant,
                                'pos': (mpos[0] + self.scroll[0], mpos[1] + self.scroll[1])
                            })
                    if event.button == 3:
                        self.right_clicking = True
                    if self.shift:
                        if event.button == 4:
                            self.tile_variant = (self.tile_variant - 1) % len(self.assets[self.tile_list[self.tile_group]])
                        if event.button == 5:
                            self.tile_variant = (self.tile_variant + 1) % len(self.assets[self.tile_list[self.tile_group]])
                    else:
                        if event.button == 4:
                            self.tile_group = (self.tile_group - 1) % len(self.tile_list)
                            self.tile_variant = 0
                        if event.button == 5:
                            self.tile_group = (self.tile_group + 1) % len(self.tile_list)
                            self.tile_variant = 0
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.clicking = False
                    if event.button == 3:
                        self.right_clicking = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.movement[0] = True
                    if event.key == pygame.K_d:
                        self.movement[1] = True
                    if event.key == pygame.K_w:
                        self.movement[2] = True
                    if event.key == pygame.K_s:
                        self.movement[3] = True
                    if event.key == pygame.K_g:
                        self.ongrid = not self.ongrid
                    if event.key == pygame.K_t:
                        self.tilemap.autotile()
                    if event.key == pygame.K_o:
                        self.tilemap.save('map.json')
                    if event.key == pygame.K_LSHIFT:
                        self.shift = True
                        self.scroll_speed += 2
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        self.movement[0] = False
                    if event.key == pygame.K_d:
                        self.movement[1] = False
                    if event.key == pygame.K_w:
                        self.movement[2] = False
                    if event.key == pygame.K_s:
                        self.movement[3] = False
                    if event.key == pygame.K_LSHIFT:
                        self.shift = False
                        self.scroll_speed = 3

            # Scale and render display onto the main screen.
            scaled_display = pygame.transform.scale(self.display, self.screen.get_size())
            self.screen.blit(scaled_display, (0, 0))
            pygame.display.update()
            self.clock.tick(60)

# ---------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    editor = Editor()
    editor.run()

import argparse
import os
import sys
import pygame
import time

# Reuse the game's rendering/look by importing PongGame
from gameplay.pong_game import PongGame, SCREEN_WIDTH, SCREEN_HEIGHT
# Use the HDF5 reader utilities from the simulation module
from pong_simulation.pong_simulation import PongH5Reader


def replay_pong(h5_path: str, fps: int = 60):
    """
    Replay the Pong game using positions recorded in an HDF5 file.

    Parameters:
    - h5_path: Path to the HDF5 file produced by pong_simulation.
    - fps: Frames per second for replay.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    # Load time series lazily
    with PongH5Reader(h5_path) as data:
        ball_ds = data.ball_pos  # shape (T, 2)
        plat_ds = data.platform_pos  # shape (T,)
        num_frames = min(ball_ds.shape[0], plat_ds.shape[0])

        # Initialize pygame and the game renderer
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pong Replay")
        clock = pygame.time.Clock()

        # logging=False to avoid CSV side effects during replay
        game = PongGame(SCREEN_WIDTH, SCREEN_HEIGHT, logging=False)

        running = True
        t = 0
        while running and t < num_frames:
            # Handle minimal events to keep window responsive and allow quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            

            # Apply recorded positions for this frame
            bx, by = ball_ds[t]
            py = plat_ds[t]

            # Update game object's state directly (no physics step)
            game.ball.x = int(bx)
            game.ball.y = int(by)
            game.set_platform_position(int(py))

            # Draw current frame using the game's draw routine
            game.draw(screen)
            pygame.display.flip()

            clock.tick(fps)
            t += 1

            time.sleep(0.3)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Replay Pong from HDF5 log.")
    parser.add_argument(
        "--h5", "-i", default=os.path.join("output", "pong_simulation.h5"),
        help="Path to pong_simulation HDF5 file (default: output/pong_simulation.h5)"
    )
    parser.add_argument("--fps", type=int, default=60, help="Replay frames per second (default: 60)")
    args = parser.parse_args()

    try:
        replay_pong(args.h5, fps=args.fps)
    except Exception as e:
        print(f"Error during replay: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

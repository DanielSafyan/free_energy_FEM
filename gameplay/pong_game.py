import pygame
import sys
import csv
import os
from datetime import datetime
import time
import numpy as np

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Paddle properties
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 200
PADDLE_SPEED = 7

# Ball properties
BALL_SIZE = 15
BALL_SPEED_X = 5
BALL_SPEED_Y = 5



class PongGame:
    """
    This class encapsulates the entire game state and logic for a single-player Pong.
    It is designed to be modular, with clear functions for updating the state
    and interacting with game elements.
    """

    def __init__(self, width, height, logging):
        """
        Initializes the game state.

        Args:
            width (int): The width of the game screen.
            height (int): The height of the game screen.
        """
        self.screen_width = width
        self.screen_height = height

        # Create the player's paddle as a pygame.Rect object
        # It's placed on the left side of the screen, centered vertically.
        self.paddle = pygame.Rect(
            30, (self.screen_height - PADDLE_HEIGHT) // 2, PADDLE_WIDTH, PADDLE_HEIGHT
        )

        # Create the ball as a pygame.Rect object
        # It starts in the center of the screen.
        self.ball = pygame.Rect(
            (self.screen_width - BALL_SIZE) // 2,
            (self.screen_height - BALL_SIZE) // 2,
            BALL_SIZE,
            BALL_SIZE,
        )

        # Set the initial velocity of the ball
        self.ball_velocity_x = BALL_SPEED_X
        self.ball_velocity_y = int(np.random.choice([-5, -4, -3, 3, 4, 5]))
        if abs(self.ball_velocity_y) == 5:
            # Avoid perfectly circular/diagonal paths when |vy| == |vx|
            self.ball_velocity_x = BALL_SPEED_X - 1

        # Game state variables
        self.score = 0
        self.game_over = False
        self.font = pygame.font.Font(None, 74) # Font for displaying score and messages
        
        # Ball position tracking
        self.timestep = 0
        self.csv_file_path = "gameplay/ball_position_tracking.csv"
        self._initialize_csv_file()
        
        # Run-length encoding tracking
        self.current_block = None
        self.current_block_start_timestep = 0
        self.current_block_duration = 0

        self.logging = logging


        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (200, 200, 200)

        # Paddle properties
        self.PADDLE_WIDTH = 15
        self.PADDLE_HEIGHT = 200
        self.PADDLE_SPEED = 7

        # Ball properties
        self.BALL_SIZE = 15
        self.BALL_SPEED_X = 5
        self.BALL_SPEED_Y = 5

    def reset_ball(self):
        """Resets the ball to the center with a random direction."""
        self.ball.center = (self.screen_width / 2, self.screen_height / 2)
        # Randomize direction after reset
        self.ball_velocity_y *= -1 if pygame.time.get_ticks() % 2 == 0 else 1
        self.ball_velocity_x *= -1

    def step(self, sleep_time=0):
        """
        Evolves the game state by one time step.
        This function moves the ball, checks for collisions, and updates the score.
        It should be called once per frame.
        """
        if self.game_over:
            return

        # --- Ball Movement ---
        self.ball.x += self.ball_velocity_x
        self.ball.y += self.ball_velocity_y

        # --- Collision Detection ---
        # Ball with top/bottom walls
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.ball_velocity_y *= -1

        # Ball with right wall
        if self.ball.right >= self.screen_width:
            self.ball_velocity_x *= -1

        # Ball with left wall (Game Over)
        if self.ball.left <= 0:
            self.game_over = True

        # Ball with paddle
        if self.ball.colliderect(self.paddle):
            # Reverse horizontal direction
            self.ball_velocity_x *= -1
            # Prevent ball from getting stuck in the paddle
            self.ball.left = self.paddle.right
            self.score += 1
        
        # Increment timestep and log ball position
        self.timestep += 1

        if self.logging: 
            self._log_ball_position_to_csv()

        time.sleep(sleep_time)

    def draw(self, screen):
        """
        Draws all game elements to the screen.

        Args:
            screen (pygame.Surface): The Pygame screen surface to draw on.
        """
        # Keep window responsive even if the host loop is slow
        # (e.g., when coupled to a heavy simulation). This does not consume
        # events, unlike event.get(), so it remains compatible with the
        # standalone game loop in main().
        pygame.event.pump()

        # Fill background
        screen.fill(self.BLACK)

        # Draw the paddle and ball
        pygame.draw.rect(screen, self.WHITE, self.paddle)
        pygame.draw.ellipse(screen, self.WHITE, self.ball)

        # Draw the walls (as lines for visual effect)
        pygame.draw.aaline(screen, self.GRAY, (0, 0), (self.screen_width, 0)) # Top
        pygame.draw.aaline(screen, self.GRAY, (0, self.screen_height - 1), (self.screen_width, self.screen_height - 1)) # Bottom
        pygame.draw.aaline(screen, self.GRAY, (self.screen_width - 1, 0), (self.screen_width - 1, self.screen_height)) # Right

        # Display score
        score_text = self.font.render(str(self.score), True, self.WHITE)
        screen.blit(score_text, (self.screen_width // 2 - score_text.get_width() // 2, 20))

        # Display Game Over message if applicable
        if self.game_over:
            game_over_text = self.font.render("Game Over", True, self.WHITE)
            text_rect = game_over_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            screen.blit(game_over_text, text_rect)

    # --- Platform (Paddle) Control Functions ---
    def set_platform_position(self, y_pos):
        """
        Sets the vertical position of the platform (paddle).

        Args:
            y_pos (int): The new y-coordinate for the top of the paddle.
        """
        self.paddle.y = y_pos
        # Clamp the paddle to stay within the screen bounds
        if self.paddle.top < 0:
            self.paddle.top = 0
        if self.paddle.bottom > self.screen_height:
            self.paddle.bottom = self.screen_height

    def get_platform_position(self):
        """
        Reads the vertical position of the platform (paddle).

        Returns:
            int: The y-coordinate of the top of the paddle.
        """
        return self.paddle.y

    # --- Ball Information Functions ---
    def get_ball_position(self):
        """
        Reads the position of the ball.

        Returns:
            tuple[int, int]: A tuple (x, y) representing the top-left coordinates of the ball.
        """
        return (self.ball.x, self.ball.y)
    
    def _initialize_csv_file(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'start_timestep', 'end_timestep', 'duration', 'block_index', 'block_x', 'block_y'])
    
    def get_ball_block_position(self):
        """
        Determines which of the 6 blocks (2x3 grid) in front of the platform the ball is in.
        The grid is positioned between the platform and the right wall.
        
        Returns:
            tuple: (block_index, block_x, block_y) where:
                   - block_index: 0-5 (or -1 if outside grid)
                   - block_x: 0-1 (column index)
                   - block_y: 0-2 (row index)
        """
        ball_x, ball_y = self.get_ball_position()
        
        # Define the grid area in front of the platform
        # Platform is at x=30, width=15, so platform right edge is at x=45
        grid_left = self.paddle.right  # Start right after the platform
        grid_right = self.screen_width  # Extend to the right wall
        grid_top = 0
        grid_bottom = self.screen_height
        
        # Check if ball is within the grid area
        if (ball_x < grid_left or ball_x >= grid_right or 
            ball_y < grid_top or ball_y >= grid_bottom):
            return (-1, -1, -1)  # Ball is outside the grid
        
        # Calculate block dimensions
        block_width = (grid_right - grid_left) / 2  # 2 columns
        block_height = (grid_bottom - grid_top) / 3  # 3 rows
        
        # Determine which block the ball is in
        block_x = int((ball_x - grid_left) / block_width)
        block_y = int((ball_y - grid_top) / block_height)
        
        # Ensure we don't go out of bounds due to floating point precision
        block_x = min(block_x, 1)
        block_y = min(block_y, 2)
        
        # Calculate block index (0-5) with 0 at bottom-left, 1 middle-left, ... 5 top-right
        # Column-major from left to right, bottom to top within each column
        block_index = (2 - block_y) + 3 * block_x
        
        return (block_index, block_x, block_y)
    
    def get_ball_block_index(self):
        """
        Convenience accessor that returns only the block index for the ball's
        location within the 2x3 grid in front of the paddle. Returns 0-5, or
        -1 if the ball is outside the grid. This works regardless of the value
        of self.logging.
        """
        block_index, _, _ = self.get_ball_block_position()
        return block_index
    
    def _log_ball_position_to_csv(self):
        """Log the current ball position and block to the CSV file using run-length encoding."""
        block_index, block_x, block_y = self.get_ball_block_position()
        
        # Check if this is the first timestep or if the block has changed
        if self.current_block is None:
            # First timestep - initialize tracking
            self.current_block = (block_index, block_x, block_y)
            self.current_block_start_timestep = self.timestep
            self.current_block_duration = 1
        elif (block_index, block_x, block_y) == self.current_block:
            # Still in the same block - increment duration
            self.current_block_duration += 1
        else:
            # Block changed - log the previous block sequence
            self._write_block_sequence_to_csv()
            
            # Start tracking the new block
            self.current_block = (block_index, block_x, block_y)
            self.current_block_start_timestep = self.timestep
            self.current_block_duration = 1
    
    def _write_block_sequence_to_csv(self):
        """Write a completed block sequence to the CSV file."""
        if self.current_block is None:
            return
            
        timestamp = datetime.now().isoformat()
        block_index, block_x, block_y = self.current_block
        start_timestep = self.current_block_start_timestep
        end_timestep = start_timestep + self.current_block_duration - 1
        duration = self.current_block_duration
        
        with open(self.csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, start_timestep, end_timestep, duration, block_index, block_x, block_y])
    
    def finalize_logging(self):
        """Call this when the game ends to log the final block sequence."""
        if self.logging and self.current_block is not None:
            self._write_block_sequence_to_csv()


def main():
    """The main function to run the game."""
    pygame.init()

    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Single Player Pong")

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    LOGGING = False

    # Create an instance of the game
    game = PongGame(SCREEN_WIDTH, SCREEN_HEIGHT, LOGGING)

    # --- Main Game Loop ---
    running = True
    t = 0
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
                print(f"Game over after {t} timesteps")
            # Allow restarting the game with a key press after game over
            if event.type == pygame.KEYDOWN and game.game_over:
                if event.key == pygame.K_SPACE:
                    game = PongGame(SCREEN_WIDTH, SCREEN_HEIGHT, LOGGING) # Re-initialize the game

        # --- Player Input ---
        # Get the state of all keyboard buttons
        keys = pygame.key.get_pressed()
        if not game.game_over:
            current_paddle_y = game.get_platform_position()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                game.set_platform_position(current_paddle_y - PADDLE_SPEED)
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                game.set_platform_position(current_paddle_y + PADDLE_SPEED)

        # --- Game Logic ---
        # Update the game state for one frame
        game.step()
        t += 1
        # --- Drawing ---
        # Draw all elements onto the screen
        game.draw(screen)

        # --- Update the Display ---
        # Flip the display to show the new frame
        pygame.display.flip()

        # --- Frame Rate Control ---
        # Limit the game to 60 frames per second
        clock.tick(60)



    # --- Quit Pygame ---
    game.finalize_logging()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

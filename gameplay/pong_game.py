import pygame
import sys

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 400
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

    def __init__(self, width, height):
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
        self.ball_velocity_y = BALL_SPEED_Y

        # Game state variables
        self.score = 0
        self.game_over = False
        self.font = pygame.font.Font(None, 74) # Font for displaying score and messages

    def reset_ball(self):
        """Resets the ball to the center with a random direction."""
        self.ball.center = (self.screen_width / 2, self.screen_height / 2)
        # Randomize direction after reset
        self.ball_velocity_y *= -1 if pygame.time.get_ticks() % 2 == 0 else 1
        self.ball_velocity_x *= -1

    def step(self):
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

    def draw(self, screen):
        """
        Draws all game elements to the screen.

        Args:
            screen (pygame.Surface): The Pygame screen surface to draw on.
        """
        # Fill background
        screen.fill(BLACK)

        # Draw the paddle and ball
        pygame.draw.rect(screen, WHITE, self.paddle)
        pygame.draw.ellipse(screen, WHITE, self.ball)

        # Draw the walls (as lines for visual effect)
        pygame.draw.aaline(screen, GRAY, (0, 0), (self.screen_width, 0)) # Top
        pygame.draw.aaline(screen, GRAY, (0, self.screen_height - 1), (self.screen_width, self.screen_height - 1)) # Bottom
        pygame.draw.aaline(screen, GRAY, (self.screen_width - 1, 0), (self.screen_width - 1, self.screen_height)) # Right

        # Display sQUITcore
        score_text = self.font.render(str(self.score), True, WHITE)
        screen.blit(score_text, (self.screen_width // 2 - score_text.get_width() // 2, 20))

        # Display Game Over message if applicable
        if self.game_over:
            game_over_text = self.font.render("Game Over", True, WHITE)
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


def main():
    """The main function to run the game."""
    pygame.init()

    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Single Player Pong")

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # Create an instance of the game
    game = PongGame(SCREEN_WIDTH, SCREEN_HEIGHT)

    # --- Main Game Loop ---
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            # Allow restarting the game with a key press after game over
            if event.type == pygame.KEYDOWN and game.game_over:
                if event.key == pygame.K_SPACE:
                    game = PongGame(SCREEN_WIDTH, SCREEN_HEIGHT) # Re-initialize the game

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
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

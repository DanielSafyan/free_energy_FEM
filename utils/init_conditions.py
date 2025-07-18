import pygame
import pygame_gui
import numpy as np
import json
from matplotlib import cm
from matplotlib.colors import Normalize

class Grid:
    def __init__(self, screen, x_cells, y_cells, grid_area_rect, initial_value=0.0):
        self.screen = screen
        self.x_cells = x_cells
        self.y_cells = y_cells
        self.grid_area_rect = grid_area_rect
        self.cell_width = self.grid_area_rect.width / self.x_cells
        self.cell_height = self.grid_area_rect.height / self.y_cells
        self.values = np.full((self.x_cells, self.y_cells), float(initial_value))

    def draw(self):
        for x in range(self.x_cells):
            for y in range(self.y_cells):
                rect = pygame.Rect(self.grid_area_rect.left + x * self.cell_width,
                                   self.grid_area_rect.top + y * self.cell_height,
                                   self.cell_width, self.cell_height)
                # Color based on value
                val = self.values[x, y]
                # Normalize values and apply colormap
                norm = Normalize(vmin=0, vmax=1)
                cmap = cm.viridis
                color = [int(c * 255) for c in cmap(norm(val))[:3]]
                pygame.draw.rect(self.screen, color, rect)

    def get_cell_from_pos(self, pos):
        if not self.grid_area_rect.collidepoint(pos):
            return None
        
        local_x = pos[0] - self.grid_area_rect.left
        local_y = pos[1] - self.grid_area_rect.top
        
        grid_x = int(local_x / self.cell_width)
        grid_y = int(local_y / self.cell_height)
        
        if 0 <= grid_x < self.x_cells and 0 <= grid_y < self.y_cells:
            return (grid_x, grid_y)
        return None

    def add_gaussian(self, center_x, center_y, variance):
        x_coords = np.arange(self.x_cells)
        y_coords = np.arange(self.y_cells)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Create gaussian distribution
        gaussian = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * variance))
        self.values += gaussian
        # Clip values to be within [0, 0.9]
        self.values = np.clip(self.values, 0, 0.9)

    def paint_cell(self, cell):
        self.values[cell] = min(0.9, self.values[cell] + 0.3)


def draw_axes(screen, rect, max_x, max_y, font, color):
    """Draws x and y axes with ticks and labels."""
    # Draw axis lines
    pygame.draw.line(screen, color, (rect.left, rect.bottom), (rect.right, rect.bottom), 2)
    pygame.draw.line(screen, color, (rect.left, rect.top), (rect.left, rect.bottom), 2)

    # Draw labels
    x_label = font.render("x", True, color)
    y_label = font.render("y", True, color)
    screen.blit(x_label, (rect.centerx - x_label.get_width() // 2, rect.bottom + 20))
    screen.blit(y_label, (rect.left - 30, rect.centery - y_label.get_height() // 2))

    # Draw ticks and labels for x-axis
    for i in range(6):
        tick_val = i / 5 * max_x
        x_pos = rect.left + (i / 5) * rect.width
        pygame.draw.line(screen, color, (x_pos, rect.bottom), (x_pos, rect.bottom + 5), 2)
        label = font.render(f"{tick_val:.1f}", True, color)
        screen.blit(label, (x_pos - label.get_width() // 2, rect.bottom + 8))

    # Draw ticks and labels for y-axis
    for i in range(6):
        tick_val = i / 5 * max_y
        y_pos = rect.bottom - (i / 5) * rect.height
        pygame.draw.line(screen, color, (rect.left, y_pos), (rect.left - 5, y_pos), 2)
        label = font.render(f"{tick_val:.1f}", True, color)
        screen.blit(label, (rect.left - 15 - label.get_width(), y_pos - label.get_height() // 2))

def draw_color_bar(screen, rect, cmap, norm):
    """Draws a color bar on the screen."""
    axis_color = (255, 255, 255)
    font = pygame.font.SysFont(None, 24)

    # Title
    title_label = font.render("Concentration", True, axis_color)
    screen.blit(title_label, (rect.centerx - title_label.get_width() // 2, rect.top - 25))
    
    # Draw gradient
    y_steps = np.linspace(0, 1, rect.height)
    for i, y_val in enumerate(y_steps):
        color = [int(c * 255) for c in cmap(y_val)[:3]]
        pygame.draw.line(screen, color, (rect.left, rect.top + i), (rect.right, rect.top + i))

    # Draw labels
    max_label = font.render("1.0", True, axis_color)
    min_label = font.render("0.0", True, axis_color)
    screen.blit(max_label, (rect.right + 5, rect.top - 7))
    screen.blit(min_label, (rect.right + 5, rect.bottom - 7))

def save_initial_conditions(grid_values, nx, ny, filename="c1_initial.npz"):
    """
    Interpolates grid cell values to node values and saves them.
    """
    num_nodes_x = nx + 1
    num_nodes_y = ny + 1
    node_values = np.zeros((num_nodes_x * num_nodes_y,))

    # Interpolate grid cell values to node values
    # Simple average of surrounding cells for each node
    for i in range(num_nodes_x):
        for j in range(num_nodes_y):
            node_idx = j * num_nodes_x + i
            # Average values of cells touching this node
            neighbor_cells = []
            if i > 0 and j > 0: neighbor_cells.append(grid_values[i-1, j-1])
            if i < nx and j > 0: neighbor_cells.append(grid_values[i, j-1])
            if i > 0 and j < ny: neighbor_cells.append(grid_values[i-1, j])
            if i < nx and j < ny: neighbor_cells.append(grid_values[i, j])
            
            if neighbor_cells:
                node_values[node_idx] = np.mean(neighbor_cells)

    np.savez(filename, c1_initial=node_values)
    print(f"Saved initial conditions to {filename}")

def main(nx=32, ny=32, Lx=1.0, Ly=1.0):
    pygame.init()

    window_size = (1200, 800)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Initial Condition Setter")

    ui_manager = pygame_gui.UIManager(window_size)

    # UI layout
    ui_panel_width = 300
    ui_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((window_size[0] - ui_panel_width, 0), 
                                                                    (ui_panel_width, window_size[1])),
                                            manager=ui_manager)

    # Grid parameters
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 10), (100, 25)), text="X Cells:", manager=ui_manager, container=ui_panel)
    x_cells_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((120, 10), (150, 25)), manager=ui_manager, container=ui_panel)
    x_cells_input.set_text(str(nx))

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 40), (100, 25)), text="Y Cells:", manager=ui_manager, container=ui_panel)
    y_cells_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((120, 40), (150, 25)), manager=ui_manager, container=ui_panel)
    y_cells_input.set_text(str(ny))

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 70), (100, 25)), text="Max X:", manager=ui_manager, container=ui_panel)
    max_x_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((120, 70), (150, 25)), manager=ui_manager, container=ui_panel)
    max_x_input.set_text(str(Lx))

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 100), (100, 25)), text="Max Y:", manager=ui_manager, container=ui_panel)
    max_y_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((120, 100), (150, 25)), manager=ui_manager, container=ui_panel)
    max_y_input.set_text(str(Ly))
    
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 130), (100, 25)), text="Initial Value:", manager=ui_manager, container=ui_panel)
    initial_value_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((120, 130), (150, 25)), manager=ui_manager, container=ui_panel)
    initial_value_input.set_text("0.1")

    apply_grid_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 170), (260, 30)), text="Apply Grid Settings", manager=ui_manager, container=ui_panel)

    # Gaussian parameters
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 220), (100, 25)), text="Variance:", manager=ui_manager, container=ui_panel)
    variance_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((120, 220), (150, 25)), manager=ui_manager, container=ui_panel)
    variance_input.set_text("5.0")

    gaussian_mode_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 260), (125, 30)), text="Gaussian Mode", manager=ui_manager, container=ui_panel)
    brush_mode_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((145, 260), (125, 30)), text="Brush Mode", manager=ui_manager, container=ui_panel)

    # Save button
    save_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 310), (260, 30)), text="Save Initial Conditions", manager=ui_manager, container=ui_panel)

    # Define margins and available space for the grid
    margin = {"top": 40, "bottom": 50, "left": 70, "right": 100}
    available_width = window_size[0] - ui_panel_width - margin['left'] - margin['right']
    available_height = window_size[1] - margin['top'] - margin['bottom']
    
    # This will be updated dynamically
    grid_area_rect = pygame.Rect(margin['left'], margin['top'], available_width, available_height)

    # Initial grid
    grid = Grid(screen, nx, ny, grid_area_rect, float(initial_value_input.get_text()))

    clock = pygame.time.Clock()
    is_running = True
    paint_mode = 'none'  # can be 'gaussian', 'brush', or 'none'

    while is_running:
        time_delta = clock.tick(60)/1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == apply_grid_button:
                    try:
                        x_cells = int(x_cells_input.get_text())
                        y_cells = int(y_cells_input.get_text())
                        initial_value = float(initial_value_input.get_text())
                        max_x = float(max_x_input.get_text())
                        max_y = float(max_y_input.get_text())

                        # Calculate aspect ratio and update grid rect
                        aspect_ratio = max_x / max_y
                        if available_width / aspect_ratio <= available_height:
                            new_width = available_width
                            new_height = new_width / aspect_ratio
                        else:
                            new_height = available_height
                            new_width = new_height * aspect_ratio
                        
                        # Center the new grid rect
                        new_x = margin['left'] + (available_width - new_width) / 2
                        new_y = margin['top'] + (available_height - new_height) / 2
                        grid_area_rect = pygame.Rect(new_x, new_y, new_width, new_height)

                        grid = Grid(screen, x_cells, y_cells, grid_area_rect, initial_value)
                    except (ValueError, ZeroDivisionError):
                        print("Invalid grid parameters")

                if event.ui_element == gaussian_mode_button:
                    paint_mode = 'gaussian' if paint_mode != 'gaussian' else 'none'
                
                if event.ui_element == brush_mode_button:
                    paint_mode = 'brush' if paint_mode != 'brush' else 'none'
                
                if event.ui_element == save_button:
                    try:
                        max_x = float(max_x_input.get_text())
                        max_y = float(max_y_input.get_text())
                        save_initial_conditions(grid.values, grid.x_cells, grid.y_cells)
                        is_running = False # Close UI after saving
                    except ValueError:
                        print("Invalid max coordinates")

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cell = grid.get_cell_from_pos(event.pos)
                if cell:
                    if paint_mode == 'gaussian':
                        try:
                            variance = float(variance_input.get_text())
                            grid.add_gaussian(cell[0], cell[1], variance)
                        except ValueError:
                            print("Invalid variance for Gaussian")
                    elif paint_mode == 'brush':
                        grid.paint_cell(cell)

            ui_manager.process_events(event)

        ui_manager.update(time_delta)

        screen.fill((50, 50, 50))
        grid.draw()

        # Draw axes and color bar
        font = pygame.font.SysFont(None, 24)
        axis_color = (255, 255, 255)
        max_x = float(max_x_input.get_text())
        max_y = float(max_y_input.get_text())
        draw_axes(screen, grid_area_rect, max_x, max_y, font, axis_color)

        color_bar_rect = pygame.Rect(grid_area_rect.right + 50, grid_area_rect.top, 20, grid_area_rect.height)
        draw_color_bar(screen, color_bar_rect, cm.viridis, Normalize(vmin=0, vmax=1))

        ui_manager.draw_ui(screen)

        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    main()

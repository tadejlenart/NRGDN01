import struct
import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
import time

# Global variables
s = 1.0  # Initial scaling factor
scaling_factor_step = 0.1  # Step size for adjusting scaling factor
viewport_width = 1200
viewport_height = 800
aspect_ratio = viewport_width / viewport_height
filename = "plush.splat"

# Measurement variables
sorting_time = 0
render_time = 0
runtime = 0

# Vertex shader code
VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
uniform mat4 MVP;

void main(){
    gl_Position = MVP * vec4(vertexPosition_modelspace,1);
}
"""

# Fragment shader code
FRAGMENT_SHADER = """
#version 330 core

out vec4 FragColor;
uniform vec4 splatColor;
uniform float alpha;

void main(){
    // Calculate distance from the center of the splat
    float distance = length(gl_PointCoord - vec2(0.5));

    // Calculate Gaussian falloff using a Gaussian function
    float sigma = 0.5; // Adjust this value for the desired falloff
    float exponent = -0.5 * (distance / sigma) * (distance / sigma);
    float gaussianAlpha = exp(exponent);

    // Blend the splat color with the background color based on the alpha value
    FragColor = vec4(splatColor.rgb, alpha * gaussianAlpha);
}
"""


# Main loop
def main():
    global sorting_time, render_time, runtime
    start_time = time.time()  # Start measuring runtime

    pygame.init()
    display = (viewport_width, viewport_height)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)

    points = read_points_from_file("files/" + filename)

    sorting_start_time = time.time()
    transformed_points = transform_points(points, display)
    transformed_points.sort(key=lambda x: x[0][2], reverse=True)  # Sort points by depth
    sorting_end_time = time.time()
    sorting_time = sorting_end_time - sorting_start_time

    init(display, transformed_points)

    while True:
        handle_events()
        glClear(GL_COLOR_BUFFER_BIT)
        modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        render_start_time = time.time()
        draw_points(transformed_points, modelview_matrix)
        render_end_time = time.time()
        render_time = render_end_time - render_start_time
        pygame.display.flip()
        runtime = time.time() - start_time
        print("File: " + filename)
        print("Points in cloud: " + str(len(transformed_points)))
        print("Sorting Time:", sorting_time, "seconds")
        print("Render Time:", render_time, "seconds")
        print("Runtime:", runtime, "seconds")


# Function to read points from a binary file
def read_points_from_file(filename):
    points = []
    with open(filename, 'rb') as file:
        while True:
            data = file.read(32)  # Each splat is 32 bytes
            if not data:
                break
            # Unpack data according to the format specified
            point = struct.unpack('3f3f4B4B', data)
            position = point[:3]
            scale = point[3:6]
            color = [color / 255.0 for color in point[6:10]]
            rotation = list(point[10:])
            rotation[0] = (rotation[0] - 128) / 128
            rotation[1] = (rotation[1] - 128) / 128
            rotation[2] = (rotation[2] - 128) / 128
            rotation[3] = (rotation[3] - 128) / 128
            points.append((position, scale, color, tuple(rotation)))
    return points


# Function to transform points
def transform_points(points, display):
    out_points = []
    for position, scale, color, rotation in points:
        screen_pos = world_to_screen(position, display)
        out_points.append((screen_pos, scale, color, rotation, position))
    return out_points


# Function to transform world coordinates to screen coordinates
def world_to_screen(position, display):
    modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    screen_x, screen_y, screen_z = gluProject(*position, modelview_matrix, projection_matrix, viewport)
    screen_x = screen_x / display[0]
    screen_y = (screen_y / display[1]) * -1
    screen_z = screen_z
    return screen_x, screen_y, screen_z


# Function to initialize OpenGL
def init(display, points):
    # Calculate the bounding box of all points to determine the center
    min_x = min(point[0][0] for point in points)
    max_x = max(point[0][0] for point in points)
    min_y = min(point[0][1] for point in points)
    max_y = max(point[0][1] for point in points)
    min_z = min(point[0][2] for point in points)
    max_z = max(point[0][2] for point in points)

    # Calculate the center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2

    # Set the eye position to be relatively close to the center of the point cloud
    eye_x = center_x
    eye_y = center_y
    eye_z = max_z + 1.5  # Adjust this value to move the camera closer or further away

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(eye_x, eye_y, eye_z, center_x, center_y, center_z, 0, 1, 0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)

    # Compile and link shaders
    vertex_shader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

    # Check compilation status of vertex shader
    vertex_compiled = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if vertex_compiled != GL_TRUE:
        error = glGetShaderInfoLog(vertex_shader)
        print("Vertex shader compilation failed:", error)

    # Check compilation status of fragment shader
    fragment_compiled = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if fragment_compiled != GL_TRUE:
        error = glGetShaderInfoLog(fragment_shader)
        print("Fragment shader compilation failed:", error)

    # Linking status
    linked = shaders.glGetProgramiv(shader_program, GL_LINK_STATUS)
    if linked != GL_TRUE:
        error = shaders.glGetProgramInfoLog(shader_program)
        print("Shader program linking failed:", error)
    else:
        print("Shader program linking successful.")

    glUseProgram(shader_program)

    # Get uniform locations
    MVP_location = glGetUniformLocation(shader_program, "MVP")
    global splatColor_location
    splatColor_location = glGetUniformLocation(shader_program, "splatColor")
    global alpha_location
    alpha_location = glGetUniformLocation(shader_program, "alpha")

    # Calculate the Model-View-Projection (MVP) matrix
    modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
    MVP_matrix = np.dot(modelview_matrix, projection_matrix)

    # Pass the MVP matrix to the shader
    glUniformMatrix4fv(MVP_location, 1, GL_FALSE, MVP_matrix)


# Function to handle Pygame events
def handle_events():
    global s
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                s += scaling_factor_step
            elif event.key == pygame.K_DOWN:
                s -= scaling_factor_step


# Function to draw points with Gaussian falloff
def draw_points(transformed_points, modelview_matrix):
    global s, viewport_width, viewport_height, aspect_ratio
    # Clear the color buffer with white color and set alpha to 1.0
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    # Initialize destination color RGBd to white
    RGBd = (1.0, 1.0, 1.0)

    for screen_pos, scale, color, rotation, position in transformed_points:
        z = get_depth_in_view_space(position, modelview_matrix)
        final_scale = s / z
        side_length = 2 * final_scale * aspect_ratio

        side_length_normalized_x = side_length / viewport_width
        side_length_normalized_y = side_length / viewport_height

        # Calculate distance from the center of the splat
        distance_x = screen_pos[0]
        distance_y = screen_pos[1]
        distance_z = screen_pos[2]
        distance = np.sqrt(distance_x ** 2 + distance_y ** 2 + distance_z ** 2)

        # Initialize covariance matrix with uniform scaling Σ = s/z
        covariance_matrix = np.eye(3) * (s / z)

        alpha = gaussian_falloff(distance, covariance_matrix)  # Calculate Gaussian falloff

        # Apply blending
        RGBd = (
            (1 - alpha) * RGBd[0] + alpha * color[0],
            (1 - alpha) * RGBd[1] + alpha * color[1],
            (1 - alpha) * RGBd[2] + alpha * color[2]
        )

        # Set OpenGL blending mode to enable alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glPushMatrix()
        glUniform4f(splatColor_location, RGBd[0], RGBd[1], RGBd[2], alpha)
        glColor4f(RGBd[0], RGBd[1], RGBd[2], alpha)

        # Set uniform alpha value for the fragment shader
        glUniform1f(alpha_location, alpha)

        glBegin(GL_QUADS)
        glVertex3f(screen_pos[0] - side_length_normalized_x / 2, screen_pos[1] - side_length_normalized_x / 2,
                   screen_pos[2])
        glVertex3f(screen_pos[0] + side_length_normalized_x / 2, screen_pos[1] - side_length_normalized_x / 2,
                   screen_pos[2])
        glVertex3f(screen_pos[0] + side_length_normalized_x / 2, screen_pos[1] + side_length_normalized_x / 2,
                   screen_pos[2])
        glVertex3f(screen_pos[0] - side_length_normalized_x / 2, screen_pos[1] + side_length_normalized_x / 2,
                   screen_pos[2])

        glEnd()
        glPopMatrix()
        # Reset OpenGL blending mode
        glDisable(GL_BLEND)
        RGBd = (1.0, 1.0, 1.0)


# Function to get depth in view space
def get_depth_in_view_space(position, modelview_matrix):
    eye_coordinates = np.dot(modelview_matrix, [position[0], position[1], position[2], 1.0])
    return eye_coordinates[2]  # Extracting z-coordinate


# Function to calculate Gaussian falloff
def gaussian_falloff(distance, covariance_matrix):
    inverse_covariance = np.linalg.inv(covariance_matrix)
    exponent = -0.5 * np.dot(distance, np.dot(inverse_covariance, distance))
    alpha = np.exp(exponent)
    return np.trace(alpha)


if __name__ == "__main__":
    main()
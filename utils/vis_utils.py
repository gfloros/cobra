import bpy
import mathutils
import numpy as np

def set_world_background(color=(1, 1, 1)):
    """Set the world background color."""
    world = bpy.context.scene.world
    world.use_nodes = True
    background_node = world.node_tree.nodes.get('Background')
    background_node.inputs[0].default_value = (*color, 1)  # RGB + Alpha

def add_white_cube(size=10, location=(0, 0, 0)):
    # Create a cube
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    
    # Get the newly created cube object
    cube = bpy.context.object
    cube.name = "WhiteCube"

    # Create a new material for the cube
    white_material = bpy.data.materials.new(name="WhiteMaterial")
    white_material.use_nodes = True

    # Clear existing nodes
    nodes = white_material.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    # Create a new shader node for pure white
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)  # Pure white
    bsdf.inputs['Roughness'].default_value = 0.5  # Set roughness for softer shadows

    # Create an output node and connect it to the BSDF
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links = white_material.node_tree.links
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Assign the material to the cube
    if cube.data.materials:
        cube.data.materials[0] = white_material
    else:
        cube.data.materials.append(white_material)

    # Enable backface culling (optional)
    bsdf.inputs['Alpha'].default_value = 1.0  # Ensuring full opacity
    white_material.shadow_method = 'NONE'  # Ensure the cube does not cast its own shadows

    return cube


def rotate_points(points, angle_x, angle_y, angle_z):
    """
    Rotate points around the x, y, and z axes by given angles.

    Parameters:
    points (array-like): An array of points (shape: (n, 3)).
    angle_x (float): Angle to rotate around the X-axis in radians.
    angle_y (float): Angle to rotate around the Y-axis in radians.
    angle_z (float): Angle to rotate around the Z-axis in radians.

    Returns:
    np.ndarray: The rotated points.
    """
    # Convert angles from degrees to radians
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    # Define rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])

    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])

    R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])

    # Combine the rotations: first R_z, then R_y, then R_x
    R = R_z @ R_y @ R_x

    # Apply the rotation to each point
    rotated_points = np.dot(points, R.T)  # Transpose the rotation matrix for proper multiplication

    return rotated_points
def add_empty_at_collection_center(collection_name="SphereCollection"):
    """Create an empty object at the center of the specified collection to use as a tracking target."""
    # Create an empty object
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    empty = bpy.context.object
    empty.name = "CollectionCenter"
    
    # Calculate the center of the collection (average position of all spheres)
    collection = bpy.data.collections[collection_name]
    num_objects = len(collection.objects)
    
    center_x = sum(obj.location.x for obj in collection.objects) / num_objects
    center_y = sum(obj.location.y for obj in collection.objects) / num_objects
    center_z = sum(obj.location.z for obj in collection.objects) / num_objects
    
    empty.location = (center_x, center_y, center_z)  # Position the empty at the center of the collection
    
    return empty

def setup_circular_camera_path(radius=5,
                               location=(0, 0, 0),
                               init_camera_position=(0, 5, 0),
                               init_camera_rotation=(0, 0, 0),
                               path_duration=360,
                               target_object_name="CollectionCenter"):  # Add the target object name
    # Create the circular path for the camera to follow
    bpy.ops.curve.primitive_nurbs_circle_add(radius=radius, location=location)
    circle = bpy.context.object
    circle.name = "CameraPath"

    # Select the camera and clear its parent if any
    camera = bpy.data.objects["Camera"]
    camera.select_set(True)
    bpy.ops.object.parent_clear(type="CLEAR")

    # Set the camera's initial location and rotation
    camera.location = init_camera_position
    camera.rotation_euler = init_camera_rotation

    # Add a Follow Path constraint to the camera to follow the circular path
    follow_path_constraint = camera.constraints.new(type='FOLLOW_PATH')
    follow_path_constraint.target = circle
    follow_path_constraint.use_fixed_location = True

    # Set the duration of the path animation
    circle.data.path_duration = path_duration

    # Ensure the camera moves along the path over time (keyframes)
    follow_path_constraint.offset_factor = 0.0  # Start at the beginning of the path
    follow_path_constraint.keyframe_insert(data_path="offset_factor", frame=1)  # Keyframe at frame 1

    follow_path_constraint.offset_factor = 1.0  # End at the full length of the path
    follow_path_constraint.keyframe_insert(data_path="offset_factor", frame=path_duration)  # Keyframe at the last frame

    # Ensure the camera is always looking at the object (Track To constraint)
    target_object = bpy.data.objects[target_object_name]  # Replace with your object's name
    track_to_constraint = camera.constraints.new(type='TRACK_TO')
    track_to_constraint.target = target_object
    track_to_constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Camera faces the negative Z axis
    track_to_constraint.up_axis = 'UP_Y'  # Camera's up axis is the Y axis



def move_camera_closer(distance):
    """Move the camera closer to the object along its local Z-axis."""
    camera = bpy.context.scene.camera
    direction = camera.matrix_world.to_quaternion() @ mathutils.Vector(
        (0.0, 0.0, -1.0)
    )  # Local Z-axis in world coordinates
    camera.location += direction * distance


def move_camera_up_down(distance):
    """Move the camera closer to the object along its local Y-axis."""
    camera = bpy.context.scene.camera
    direction = camera.matrix_world.to_quaternion() @ mathutils.Vector(
        (0.0, 1.0, 0.0)
    )  # Local Z-axis in world coordinates
    camera.location += direction * distance


def move_camera_left_right(distance):
    """Move the camera closer to the object along its local Y-axis."""
    camera = bpy.context.scene.camera
    direction = camera.matrix_world.to_quaternion() @ mathutils.Vector(
        (1.0, 0.0, 0.0)
    )  # Local Z-axis in world coordinates
    camera.location += direction * distance


def clear_scene_objects(exceptions=None):
    """Delete all objects in the scene except those in exceptions list or lights."""
    if exceptions is None:
        exceptions = []

    bpy.ops.object.select_all(action="DESELECT")

    for obj in bpy.context.scene.objects:
        if obj.name not in exceptions and obj.type not in {"LIGHT", "CAMERA"}:
            obj.select_set(True)

    bpy.ops.object.delete()

def create_sphere_pool(num_spheres, radius=0.005):  # 0.006 for normal
    """Create a pool of spheres to be reused."""
    sphere_pool = []

    # Create a new collection to hold the spheres
    sphere_collection = bpy.data.collections.new("SphereCollection")
    bpy.context.scene.collection.children.link(sphere_collection)

    # Create the base sphere object once
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)
    base_sphere = bpy.context.object
    base_sphere.name = "BaseSphere"

    # Create a new material for the sphere
    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0,0.8,1,1)  # Set to specified color
    base_sphere.data.materials.append(mat)

    # Shade smooth for better appearance
    bpy.ops.object.shade_smooth()

    # Hide the base sphere (it will not be rendered)
    base_sphere.hide_set(True)
    base_sphere.hide_render = True

    # Duplicate spheres to create the pool
    for i in range(num_spheres):
        new_sphere = base_sphere.copy()
        new_sphere.data = base_sphere.data  # Link data block to new sphere
        bpy.context.collection.objects.link(new_sphere)
        new_sphere.location = (0, 0, 0)  # Place them all initially at the origin
        sphere_pool.append(new_sphere)

        # Add each new sphere to the collection
        sphere_collection.objects.link(new_sphere)

    return sphere_pool

def place_spheres(points, sphere_pool):
    """Translate the spheres to the specified points."""
    num_points = len(points)
    num_spheres = len(sphere_pool)

    # For each point, place a sphere
    for i in range(min(num_points, num_spheres)):
        sphere_pool[i].location = (points[i][0], points[i][1], points[i][2])
        sphere_pool[i].hide_set(False)
        sphere_pool[i].hide_render = False

    # Hide any extra spheres that are not needed
    for i in range(num_points, num_spheres):
        sphere_pool[i].hide_set(True)
        sphere_pool[i].hide_render = True

def setup_gpu_rendering():
    """Set up GPU rendering and suppress log output."""
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"

    # Select the GPU device for rendering
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        device.use = True

    bpy.context.scene.cycles.samples = 128  # Adjust based on quality/performance needs
    bpy.app.debug_value = 0  # Minimize logging output
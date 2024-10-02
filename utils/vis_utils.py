import bpy
import mathutils


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

def create_sphere_pool(num_spheres, radius=0.006):  # 0.006 for normal
    """Create a pool of spheres to be reused."""
    sphere_pool = []

    # Create the base sphere object once
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)
    base_sphere = bpy.context.object
    base_sphere.name = "BaseSphere"

    # Add material to the base sphere
    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.diffuse_color = (0, 1, 1, 1)  # Set color
    base_sphere.data.materials.append(mat)
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
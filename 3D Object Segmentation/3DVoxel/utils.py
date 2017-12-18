
import numpy as np
from IPython.display import IFrame
from IPython.display import display 
from matplotlib import pyplot as plt
import os, pdb 
import h5py 
import time, os, sys 

TEMPLATE_POINTS = """
<!DOCTYPE html>
<head>

<title>PyntCloud</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
<style>
body {{
    color: #cccccc;
    font-family: Monospace;
    font-size: 13px;
    text-align: center;
    background-color: #050505;
    margin: 0px;
    overflow: hidden;
}}
#logo_container {{
    position: absolute;
    top: 0px;
    width: 100%;
}}
.logo {{
    max-width: 20%;
}}
</style>

</head>
<body>

<div>
    <img class="logo" src="https://media.githubusercontent.com/media/daavoo/pyntcloud/master/docs/data/pyntcloud.png">
</div>

<div id="container">
</div>

<script src="http://threejs.org/build/three.js"></script>
<script src="http://threejs.org/examples/js/Detector.js"></script>
<script src="http://threejs.org/examples/js/controls/OrbitControls.js"></script>
<script src="http://threejs.org/examples/js/libs/stats.min.js"></script>

<script>

    if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

    var container, stats;
    var camera, scene, renderer;
    var points;

    init();
    animate();

    function init() {{

        var camera_x = {camera_x};
        var camera_y = {camera_y};
        var camera_z = {camera_z};
        
        var look_x = {look_x};
        var look_y = {look_y};
        var look_z = {look_z};

        var positions = new Float32Array({positions});

        var colors = new Float32Array({colors});

        var points_size = {points_size};

        var axis_size = {axis_size};

        container = document.getElementById( 'container' );

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera( 90, window.innerWidth / window.innerHeight, 0.1, 1000 );
        camera.position.x = camera_x;
        camera.position.y = camera_y;
        camera.position.z = camera_z;
        camera.up = new THREE.Vector3( 0, 0, 1 );       

        if (axis_size > 0){{
            var axisHelper = new THREE.AxisHelper( axis_size );
            scene.add( axisHelper );
        }}

        var geometry = new THREE.BufferGeometry();
        geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
        geometry.addAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );
        geometry.computeBoundingSphere();

        var material = new THREE.PointsMaterial( {{ size: points_size, vertexColors: THREE.VertexColors }} );

        points = new THREE.Points( geometry, material );
        scene.add( points );


        renderer = new THREE.WebGLRenderer( {{ antialias: false }} );
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( window.innerWidth, window.innerHeight );

        controls = new THREE.OrbitControls( camera, renderer.domElement );
        controls.target.copy( new THREE.Vector3(look_x, look_y, look_z) );
        camera.lookAt( new THREE.Vector3(look_x, look_y, look_z));

        container.appendChild( renderer.domElement );

        window.addEventListener( 'resize', onWindowResize, false );
    }}

    function onWindowResize() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize( window.innerWidth, window.innerHeight );
    }}

    function animate() {{
        requestAnimationFrame( animate );
        render();
    }}

    function render() {{
        renderer.render( scene, camera );
    }}
</script>

</body>
</html>
"""


TEMPLATE_VG = """
<!DOCTYPE html>
<head>

<title>PyntCloud</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
<style>
    body {{
        color: #cccccc;font-family: Monospace;
        font-size: 13px;
        text-align: center;
        background-color: #050505;
        margin: 0px;
        overflow: hidden;
    }}
    #logo_container {{
        position: absolute;
        top: 0px;
        width: 100%;
    }}
    .logo {{
        max-width: 20%;
    }}
</style>

</head>
<body>

<div>
    <img class="logo" src="https://media.githubusercontent.com/media/daavoo/pyntcloud/master/docs/data/pyntcloud.png">
</div>

<div id="container">
</div>

<script src="http://threejs.org/build/three.js"></script>
<script src="http://threejs.org/examples/js/Detector.js"></script>
<script src="http://threejs.org/examples/js/controls/OrbitControls.js"></script>
<script src="http://threejs.org/examples/js/libs/stats.min.js"></script>

<script>

    if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

    var container, stats;
    var camera, scene, renderer;
    var points;

    init();
    animate();

    function init() {{

        var camera_x = {camera_x};
        var camera_y = {camera_y};
        var camera_z = {camera_z};

        var look_x = {look_x};
        var look_y = {look_y};
        var look_z = {look_z};

        var X = new Float32Array({X});
        var Y = new Float32Array({Y});
        var Z = new Float32Array({Z});

        var R = new Float32Array({R});
        var G = new Float32Array({G});
        var B = new Float32Array({B});

        var S_x = {S_x};
        var S_y = {S_y};
        var S_z = {S_z};

        var n_voxels = {n_voxels};
        var axis_size = {axis_size};

        container = document.getElementById( 'container' );

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera( 90, window.innerWidth / window.innerHeight, 0.1, 1000 );
        camera.position.x = camera_x;
        camera.position.y = camera_y;
        camera.position.z = camera_z;
        camera.up = new THREE.Vector3( 0, 0, 1 );   

        if (axis_size > 0){{
            var axisHelper = new THREE.AxisHelper( axis_size );
            scene.add( axisHelper );
        }}

        var geometry = new THREE.BoxGeometry( S_x, S_z, S_y );

        for ( var i = 0; i < n_voxels; i ++ ) {{            
            var mesh = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial() );
            mesh.material.color.setRGB(R[i], G[i], B[i]);
            mesh.position.x = X[i];
            mesh.position.y = Y[i];
            mesh.position.z = Z[i];
            scene.add(mesh);
        }}
        
        renderer = new THREE.WebGLRenderer( {{ antialias: false }} );
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( window.innerWidth, window.innerHeight );

        controls = new THREE.OrbitControls( camera, renderer.domElement );
        controls.target.copy( new THREE.Vector3(look_x, look_y, look_z) );
        camera.lookAt( new THREE.Vector3(look_x, look_y, look_z));

        container.appendChild( renderer.domElement );

        window.addEventListener( 'resize', onWindowResize, false );
    }}

    function onWindowResize() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize( window.innerWidth, window.innerHeight );
    }}

    function animate() {{
        requestAnimationFrame( animate );
        render();
    }}

    function render() {{
        renderer.render( scene, camera );
    }}
</script>
</body>
</html>
"""

def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]


def plot_points(xyz, colors=None, size=0.1, axis=False):
 
    positions = xyz.reshape(-1).tolist()

    camera_position = xyz.max(0) + abs(xyz.max(0))

    look = xyz.mean(0)

    if colors is None:
        colors = [1,0.5,0] * len(positions)

    elif len(colors.shape) > 1:
        colors = colors.reshape(-1).tolist()

    if axis:
        axis_size = xyz.ptp() * 1.5
    else:
        axis_size = 0

    with open("plot_points.html", "w") as html:
        html.write(TEMPLATE_POINTS.format(
            camera_x=camera_position[0],
            camera_y=camera_position[1],
            camera_z=camera_position[2],
            look_x=look[0],
            look_y=look[1],
            look_z=look[2],
            positions=positions,
            colors=colors,
            points_size=size,
            axis_size=axis_size))

    return IFrame("plot_points.html",width=800, height=800)



def plot_voxelgrid(v_grid, cmap="Oranges", axis=False):
    scaled_shape = v_grid.shape / np.min(v_grid.shape) # shape: step size in discretization 

    # coordinates returned from argwhere are inversed so use [:, ::-1]
    # Equivalent to np.transpose(x, [2, 1, 0]) 
    points = np.argwhere(v_grid.vector)[:, ::-1] * scaled_shape

    s_m = plt.cm.ScalarMappable(cmap=cmap)
    # [:, :-1]: get rid of the last dimension. i. e (2 x 3 x 4 x 5) => (2 x 3 x 4)
    rgb = s_m.to_rgba(v_grid.vector.reshape(-1)[v_grid.vector.reshape(-1) > 0])[:,:-1]

    camera_position = points.max(0) + abs(points.max(0))
    look = points.mean(0)
    
    if axis:
        axis_size = points.ptp() * 1.5
    else:
        axis_size = 0

    with open("plotVG.html", "w") as html:
        html.write(TEMPLATE_VG.format( 
            camera_x=camera_position[0],
            camera_y=camera_position[1],
            camera_z=camera_position[2],
            look_x=look[0],
            look_y=look[1],
            look_z=look[2],
            X=points[:,0].tolist(),
            Y=points[:,1].tolist(),
            Z=points[:,2].tolist(),
            R=rgb[:,0].tolist(),
            G=rgb[:,1].tolist(),
            B=rgb[:,2].tolist(),
            S_x=scaled_shape[0],
            S_y=scaled_shape[2],
            S_z=scaled_shape[1],
            n_voxels=sum(v_grid.vector.reshape(-1) > 0),
            axis_size=axis_size))

    return IFrame("plotVG.html",width=800, height=800)


def disp_image(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()



# data_path = "../data/3d-mnist"
# train_dir = os.path.join(data_path, 'train_point_clouds.h5')
 
# with h5py.File(train_dir, 'r') as hf:
#     zero = hf["0"]
#     digit_zero = (zero["img"][:], zero["points"][:], zero.attrs["label"])
#     pc_zero = digit_zero[1]  # Numpy, 25700 x 3 
#     lab_zero = digit_zero[2] # Numpy int64 
#     img_zero = digit_zero[0] # Numpy 30 x 30, float64 
#     # disp_image(img_zero)
#     iframe = plot_points(pc_zero)
#     display(iframe)


########################################################
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = [] 
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    return tot_time  

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
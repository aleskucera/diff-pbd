from brax.positional import pipeline
from brax.io import mjcf
# @title Colab setup and imports
import brax
import jax
from jax import numpy as jp
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

ball = mjcf.loads(
    """
    <mujoco>
        <option timestep="0.005"/>
        <worldbody>
            <body pos="0 0 3">
                <joint type="free"/>
                <geom size="0.5" type="sphere"/>
            </body>
            <geom size="40 40 40" type="plane"/>
        </worldbody>
   </mujoco>
  """
)


def visualize(ax, pos, alpha=1):
    for p, pn in zip(pos, list(pos[1:]) + [None]):
        ax.add_patch(Circle(xy=(p[0], p[2]), radius=0.5, fill=False, color=(0, 0, 0, alpha)))
        if pn is not None:
            ax.add_line(Line2D([p[0], pn[0]], [p[2], pn[2]], color=(1, 0, 0, alpha)))

elasticity = 0.85  # @param { type:"slider", min: 0.5, max: 1.0, step:0.05 }
ball_velocity = 1  # @param { type:"slider", min:-5, max:5, step: 0.5 }

# change the material elasticity of the ball and the plane
ball = ball.replace(elasticity=jp.array([elasticity] * ball.ngeom))

# provide an initial velocity to the ball
qd = jp.array([ball_velocity, 0, 0, 0, 0, 0])
state = jax.jit(pipeline.init)(ball, ball.init_q, qd)

_, ax = plt.subplots()
plt.xlim([-3, 3])
plt.ylim([0, 4])

for i in range(1000):
    if i % 10 == 0:
        visualize(ax, state.x.pos, i / 1000.)
    state = jax.jit(pipeline.step)(ball, state, None)

plt.title('ball in motion')
plt.show()

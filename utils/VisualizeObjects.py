import pyvista
import numpy as np

def visualize_dif_obj(file):
    points = np.loadtxt(file)
    print(f"Количество точек {points.shape[1]} - мерного изделия:", len(points))
    plotter = pyvista.Plotter(window_size=(1000, 700))
    plotter.add_points(points[::1], color='blue', point_size=3, label='Объект (точки)')
    plotter.add_legend()
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show(interactive=True)

if __name__ == '__main__':
    file = 'teapot.txt'
    visualize_dif_obj(file)
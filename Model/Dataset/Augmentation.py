import numpy as np
import torch
from torch.utils.data import Dataset

class ObjectAugment:
    def __init__(self, points):
        self.points = points

    def random_rotation_3d(self):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        psi = np.random.uniform(0, 2*np.pi)
        Rx = np.array([[1,0,0],
                    [0,np.cos(theta),-np.sin(theta)],
                    [0,np.sin(theta),np.cos(theta)]])
        Ry = np.array([[np.cos(phi),0,np.sin(phi)],
                    [0,1,0],
                    [-np.sin(phi),0,np.cos(phi)]])
        Rz = np.array([[np.cos(psi),-np.sin(psi),0],
                    [np.sin(psi),np.cos(psi),0],
                    [0,0,1]])
        R = Rz @ Ry @ Rx
        self.points = self.points @ R.T

    def random_scale(self, scale_range=(0.99, 1.01)):
        scale = np.random.uniform(*scale_range)
        self.points = self.points * scale

    def random_translate(self, shift_range=0.01):
        shift = np.random.uniform(-shift_range, shift_range, size=3)
        self.points = self.points + shift

    def add_noise(self, noise_std=0.01):
        noise = np.random.normal(0, noise_std, size=self.points.shape)
        self.points = self.points + noise

    def full_augment(self):

        self.random_rotation_3d()
        self.random_scale()
        self.random_translate()
        self.add_noise()

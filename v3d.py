from typing import Tuple, Optional
import math
import sys


Vector3d = Tuple[float, float, float]


def create(x: float, y: float, z: float) -> Vector3d:
    return x, y, z


def add(a: Vector3d, b: Vector3d) -> Vector3d:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vector3d, b: Vector3d) -> Vector3d:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def dot(a: Vector3d, b: Vector3d) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def neg(a: Vector3d) -> Vector3d:
    return - a[0], - a[1], - a[2]


def cross(a: Vector3d, b: Vector3d) -> Vector3d:
    return a[1] * b[2] - a[2] * b[1], \
           a[2] * b[0] - a[0] * b[2], \
           a[0] * b[1] - a[1] * b[0]


def mul(a: Vector3d, b: float) -> Vector3d:
    return a[0] * b, a[1] * b, a[2] * b


def squaredLength(a: Vector3d) -> float:
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]


def length(a: Vector3d) -> float:
    return math.sqrt(squaredLength(a))


def normalize(a: Vector3d) -> Optional[Vector3d]:
    a_length = length(a)
    if a_length < sys.float_info.epsilon:
        return 0, 0, 0
    return mul(a, 1.0 / a_length)

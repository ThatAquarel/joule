import glm


class RightHandedAxes:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._transform = glm.mat4x4(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def get_right_handed(self):
        return self._transform

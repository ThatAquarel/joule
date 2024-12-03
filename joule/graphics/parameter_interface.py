import imgui
from imgui.integrations.glfw import GlfwRenderer

import numpy as np


def ui_spacing():
    for _ in range(5):
        imgui.spacing()


def ui_section(section_name, top_margin=True):
    def decorator(func):
        def wrapper():
            if top_margin:
                ui_spacing()

            imgui.text(section_name)
            imgui.separator()

            func()

        return wrapper

    return decorator


class ParameterInterface:
    def __init__(self, window):
        imgui.create_context()

        self._imgui_impl = GlfwRenderer(window, attach_callbacks=False)

        self.expression_textbox = ""
        self.parser_label = ""
        self.x_domain_slider = [-np.pi, np.pi]
        self.y_domain_slider = [-np.pi, np.pi]

        self.mass_slider = 10.0
        self.gravity_slider = 25.0
        self.friction_slider = 0.2
        self.z_correction = True

        self.ball_color = [0.25, 0.25, 0.25]
        self.surface_color = [1.0, 1.0, 1.0]
        self.light_color = [1.0, 1.0, 1.0]
        self.background_color

        self.ambient_strength = 0.2
        self.diffuse_strength = 0.6
        self.diffuse_base = 0.5
        self.specular_strength = 1.0
        self.specular_reflection = 16.0

    @property
    def imgui_impl(self):
        return self._imgui_impl

    @ui_section("Render", top_margin=False)
    def _render(self, dt):
        if dt:
            imgui.text(f"{1/dt:.2f} fps")
        n_bodies = self.mecanics_engine.get_render_n()
        max_bodies = self.mecanics_engine.get_render_max()
        imgui.text(f"{n_bodies}/{max_bodies} bodies")

    @ui_section("Expression")
    def _expression(self):
        window_width = imgui.get_window_width()
        _, self.expression_textbox = imgui.input_text_multiline(
            "",
            self.expression_textbox,
            1024,
            window_width - 16,
            100,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )

        # range_values = [10.0, 50.0, 10.0, 50.0]
        changed, self.x_domain_slider = imgui.slider_float2(
            "x domain",
            *self.x_domain_slider,
            -4 * np.pi,
            4 * np.pi,
        )

        self.x_domain_slider = list(self.x_domain_slider)
        x_min = np.min(self.x_domain_slider)
        x_max = np.max(self.x_domain_slider)
        self.x_domain_slider[0] = x_min
        self.x_domain_slider[1] = x_max

        changed, self.y_domain_slider = imgui.slider_float2(
            "y domain",
            *self.y_domain_slider,
            -4 * np.pi,
            4 * np.pi,
        )

        self.y_domain_slider = list(self.y_domain_slider)
        y_min = np.min(self.y_domain_slider)
        y_max = np.max(self.y_domain_slider)
        self.y_domain_slider[0] = y_min
        self.y_domain_slider[1] = y_max

        if imgui.button("Evaluate"):
            self.parser_label = self.update_function(
                self.expression_textbox,
                self.x_domain_slider,
                self.y_domain_slider,
            )
            self.expression_textbox = ""

        imgui.text(self.parser_label)

    @ui_section("Parameters")
    def _parameters(self):
        _, self.mass_slider = imgui.slider_float(
            "mass (kg)",
            self.mass_slider,
            0.0,
            100.0,
        )

        changed, self.gravity_slider = imgui.slider_float(
            "gravity (m/s^2)",
            self.gravity_slider,
            0.0,
            100.0,
        )
        if changed:
            self.mecanics_engine.set_gravity(self.gravity_slider)

        changed, self.friction_slider = imgui.slider_float(
            "fiction (k)",
            self.friction_slider,
            0.0,
            1.0,
        )
        if changed:
            self.mecanics_engine.set_friction(self.friction_slider)

        imgui.spacing()

        changed, self.ball_color[:] = imgui.color_edit3(
            "ball color",
            *self.ball_color,
        )
        if changed:
            self.balls.set_color(self.ball_color)

        changed, self.surface_color[:] = imgui.color_edit3(
            "surface color",
            *self.surface_color,
        )
        if changed:
            self.surface.set_color(self.surface_color)

    @ui_section("Functions")
    def _functions(self):
        for name, expression in self.function_texts.items():
            imgui.spacing()
            imgui.text(f"{name}\n{expression}")

    def on_render_ui(self, dt):
        imgui.new_frame()
        imgui.begin("Joule")

        self._render(dt)
        self._expression()
        self._parameters()
        self._functions()

        imgui.end()
        imgui.render()

import imgui
from imgui.integrations.glfw import GlfwRenderer

import numpy as np


def slider_domain_clamp(domain):
    """
    Clamps domain range slider values, so that
    the left slider always has the smallest value;
    the right slider, the largest value

    :param domain: Domain of slider as (min, max)
    :return: Clamped values as (min, max)
    """

    return [np.min(domain), np.max(domain)]


def ui_spacing():
    """
    Creates a blank space in the user interface
    """

    for _ in range(5):
        imgui.spacing()


def ui_section(section_name, top_margin=True):
    """
    Decorator constructor to wrap a user interface
    section with a title: reduce the repetition
    of creating a separation for every section
    """

    # constructed decorator
    def decorator(func):
        # wrap the method, building the title
        # of the section first, before drawing
        # the rest of the section

        def wrapper(*args, **kwargs):
            if top_margin:
                ui_spacing()

            # display section title
            imgui.text(section_name)
            imgui.separator()

            # draw the rest of the section
            func(*args, **kwargs)

        return wrapper

    return decorator


class ParameterInterface:
    def __init__(
        self,
        window,
        on_evaluate,
        on_change_ball_color,
        on_change_surface_color,
    ):
        """
        Parameter Interface: Manages the state of the parameters
        controlled by the user, and draws the interface

        :param window: glfw window
        :param on_evaluate: Callback for a new user function's evaluation call
        :param on_change_ball_color: Callback to change ball color
        :param on_change_surface_color: Callback to change surface color
        """

        # create DearImGui instance for ui drawing
        imgui.create_context()

        # create bindings to Glfw rendering backend
        # the application manages its own callbacks
        self._imgui_impl = GlfwRenderer(window, attach_callbacks=False)

        # ui state variables of section: Status
        self.dt = 0.0
        self.n_bodies = 0
        self.buffer_size = 0
        self.show_axes = True

        # ui state variables of section: Expression
        self.expression_textbox = "sin(x + y)"
        self.parser_response = ""
        self.x_domain_slider = [-np.pi, np.pi]
        self.y_domain_slider = [-np.pi, np.pi]

        self._on_evaluate = on_evaluate

        # ui state variables of section: Physics Parameters
        self.mass_slider = 10.0
        self.gravity_slider = 25.0
        self.friction_slider = 0.2
        self.z_correction = True

        # ui state variables of section: Render Parameters
        self.ball_color = [0.25, 0.25, 0.25]
        self._on_change_ball_color = on_change_ball_color
        self.surface_color = [1.0, 1.0, 1.0]
        self._on_change_surface_color = on_change_surface_color

        self.light_color = [1.0, 1.0, 1.0]
        self.background_color = [0.86, 0.87, 0.87]

        self.ambient_strength = 0.2
        self.diffuse_strength = 0.2
        self.diffuse_base = 0.3
        self.specular_strength = 0.1
        self.specular_reflection = 16.0

        # ui state variables of section: Differentiation Results
        self.function_texts = {}

    @property
    def want_keyboard(self):
        return imgui.get_io().want_capture_keyboard

    @property
    def want_mouse(self):
        return imgui.get_io().want_capture_mouse

    def update_status(self, dt, n_bodies, buffer_size):
        """
        Update data of section: Status

        :param dt: Time taken for frame render (s)
        :param n_bodies: Number of bodies currently rendering
        :param buffer_size: Number of bodies buffered by the physics engine
        """

        self.dt = dt
        self.n_bodies = n_bodies
        self.buffer_size = buffer_size

    def update_differentiation(self, parser_response, function_texts):
        """
        Update data of section: Expression

        :parser_response: Textual response of expression parser
        :function_texts: Textual results of differentiation
        """

        self.parser_response = parser_response
        self.function_texts = function_texts

    @property
    def impl(self):
        return self._imgui_impl

    @ui_section("Status", top_margin=False)
    def _status(self):
        """
        Draw section: Status
        """

        # dt starts off being zero, so we prevent
        # a zero division error on the first frame
        if self.dt:
            imgui.text(f"{1 / self.dt:.2f} fps")

        imgui.text(f"{self.n_bodies}/{self.buffer_size} bodies")

        _, self.show_axes = imgui.checkbox("show xyz axes", self.show_axes)

    @ui_section("Expression")
    def _expression(self):
        """
        Draw section: Expression
        """

        window_width = imgui.get_window_width()

        # user entered function expression
        _, self.expression_textbox = imgui.input_text_multiline(
            "",
            self.expression_textbox,
            1024,
            window_width - 16,
            100,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )

        # function evaluation domain sliders
        _, self.x_domain_slider = imgui.slider_float2(
            "x domain",
            *self.x_domain_slider,
            -4 * np.pi,
            4 * np.pi,
        )
        self.x_domain_slider = slider_domain_clamp(self.x_domain_slider)

        _, self.y_domain_slider = imgui.slider_float2(
            "y domain",
            *self.y_domain_slider,
            -4 * np.pi,
            4 * np.pi,
        )
        self.y_domain_slider = slider_domain_clamp(self.y_domain_slider)

        # function evaluate
        if imgui.button("Evaluate"):
            self._on_evaluate(
                self.expression_textbox,
                self.x_domain_slider,
                self.y_domain_slider,
            )

        # parser response
        imgui.text(self.parser_response)

    @ui_section("Physics Parameters")
    def _physics_parameters(self):
        """
        Draw section: Physics Parameters
        """

        # parameters for physics engine
        _, self.mass_slider = imgui.slider_float(
            "mass (kg)",
            self.mass_slider,
            0.0,
            100.0,
        )
        _, self.gravity_slider = imgui.slider_float(
            "gravity (m/s^2)",
            self.gravity_slider,
            0.0,
            100.0,
        )
        _, self.friction_slider = imgui.slider_float(
            "friction (k)",
            self.friction_slider,
            0.0,
            1.0,
        )
        _, self.z_correction = imgui.checkbox(
            "z integration correction", self.z_correction
        )

    @ui_section("Render Parameters")
    def _render_parameters(self):
        """
        Draw section: Render Parameters
        """

        # ball and surface color change are event based
        # as they need a rebuild of the vertex buffer objects
        # to change color
        ball_color_changed, self.ball_color[:] = imgui.color_edit3(
            "ball color",
            *self.ball_color,
        )
        surface_color_changed, self.surface_color[:] = imgui.color_edit3(
            "surface color",
            *self.surface_color,
        )

        # if they are updated, call their respective callbacks with
        # the new color
        if ball_color_changed:
            self._on_change_ball_color(self.ball_color)
        if surface_color_changed:
            self._on_change_surface_color(self.surface_color)

        # shader parameters
        _, self.light_color[:] = imgui.color_edit3(
            "light color",
            *self.light_color,
        )
        _, self.background_color[:] = imgui.color_edit3(
            "background color",
            *self.background_color,
        )

        imgui.spacing()

        # lighting calculation parameters
        _, self.ambient_strength = imgui.slider_float(
            "ambient: strength",
            self.ambient_strength,
            0.0,
            1.0,
        )
        _, (self.diffuse_strength, self.diffuse_base) = imgui.slider_float2(
            "diffuse: strength, base",
            self.diffuse_strength,
            self.diffuse_base,
            0.0,
            1.0,
        )
        _, self.specular_strength = imgui.slider_float(
            "specular: strength", self.specular_strength, 0.0, 1.0
        )
        _, self.specular_reflection = imgui.slider_float(
            "specular: reflection", self.specular_reflection, 0.0, 32.0
        )

    @ui_section("Differentiation Results")
    def _functions(self):
        """
        Draw section: Differentiation Results
        """

        # iterate through function and its derivatives
        # to display the symbolic expressions in their
        # textual forms
        for name, expression in self.function_texts.items():
            imgui.spacing()
            imgui.text(f"{name}\n{expression}")

    def on_render_ui(self):
        """
        Renders full user interface on frame
        """

        imgui.new_frame()
        imgui.begin("Joule")

        # draw individual sections
        self._status()
        self._expression()
        self._physics_parameters()
        self._render_parameters()
        self._functions()

        imgui.end()
        imgui.render()

# Joule

>A Dynamic Surface, Potential Energy Rolling Motion Simulator

Final Project for **Programming in Science, section 00007**

![demo](docs/demo.PNG)


## Author


Alex Xia<sup>1</sup>

<sup>1</sup>Dawson College, Montréal

![image](https://github.com/user-attachments/assets/3cb84f17-91aa-46a1-8e5e-d2c0823da192)


## Controls

The interface is more intuitive with a mouse:

- Left-click: Add ball into simulation
- Right-click drag: Rotate view
- Right-click drag + Left control: Translate view
- Scrollwheel: Zoom in/out view

The *Expression* textbox takes any function f(x, y) that can be evaluated over the selected domain.

The *Evaluate* button will plot the function entered within the *Expression* textbox


## Installation

Clone source code
```bash
git clone https://github.com/ThatAquarel/joule.git
cd joule/
```

Install requirements and install Joule as a local package. Sometimes, the installation of `imgui[glfw]` on Windows throws an error: install Visual C++ Redistributable as per the error message and then directly run `pip install imgui[glfw]`. Also, developed on Python 3.11.
```bash
pip install -r requirements.txt
pip install -e .
```

Run application
```bash
python -m joule
```

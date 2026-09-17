"""OpenGL 3.3 tabletop renderer; geometry stays subpixel until rasterization.

Physics and observations remain on the CPU. The GPU draws batched road strips,
procedural wood, filtered car textures and a cached native-resolution HUD.
"""

import numpy as np
from collections import OrderedDict
import pygame
import moderngl
from crashlearn_sim.rendering.art import car_model

VERTEX = """#version 330
in vec2 position;
in vec4 color;
uniform vec2 size;
out vec4 tint;
void main() {
    gl_Position = vec4(position / size * vec2(2,-2) + vec2(-1,1),0,1);
    tint = color;
}"""
FRAGMENT = """#version 330
in vec4 tint;
out vec4 fragment;
void main() { fragment = tint; }"""
TEXTURE_VERTEX = """#version 330
in vec2 position;
in vec2 uv;
uniform vec2 size;
out vec2 texcoord;
void main() {
    gl_Position = vec4(position / size * vec2(2,-2) + vec2(-1,1),0,1);
    texcoord = uv;
}"""
TEXTURE_FRAGMENT = """#version 330
uniform sampler2D image;
in vec2 texcoord;
out vec4 fragment;
void main() { fragment = texture(image,texcoord); }"""
WOOD_FRAGMENT = """#version 330
uniform vec2 size;
uniform vec2 origin;
uniform vec2 anchor;
uniform vec2 heading;
uniform float scale;
uniform bool overview;
out vec4 fragment;
void main() {
    vec2 q = (vec2(gl_FragCoord.x,size.y-gl_FragCoord.y)-anchor)/scale;
    vec2 p = origin + (overview ? vec2(q.x,-q.y) :
        vec2(-heading.y*q.x-heading.x*q.y,heading.x*q.x-heading.y*q.y));
    float row = floor(p.y/9.0);
    float shade = mod(row*17.0,13.0)/255.0;
    vec3 wood = vec3(139,85,58)/255.0 + shade;
    float grain = sin(p.y*35.0 + sin(p.x*.32+row)*1.7);
    wood += grain*.012;
    float seam = min(mod(p.y,9.0),9.0-mod(p.y,9.0));
    float aa = max(fwidth(p.y),.015);
    wood = mix(vec3(.35,.19,.14),wood,smoothstep(.025,.025+aa,seam));
    float joint = mod(p.x-mod(row,3.0)*12.0,38.0);
    wood *= .8+.2*smoothstep(0.0,max(fwidth(p.x),.02),min(joint,38.0-joint));
    fragment = vec4(wood,1);
}"""


class GpuScene:
    def __init__(self):
        self.ctx = moderngl.create_context(require=330)
        self.name = self.ctx.info["GL_RENDERER"]
        self.solid = self.ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.textured = self.ctx.program(
            vertex_shader=TEXTURE_VERTEX, fragment_shader=TEXTURE_FRAGMENT
        )
        self.wood = self.ctx.program(vertex_shader=TEXTURE_VERTEX, fragment_shader=WOOD_FRAGMENT)
        self.buffer = self.ctx.buffer(reserve=4 * 1024 * 1024, dynamic=True)
        self.mesh = self.ctx.vertex_array(self.solid, [(self.buffer, "2f 4f", "position", "color")])
        self.quad_buffer = self.ctx.buffer(reserve=64, dynamic=True)
        self.quad = self.ctx.vertex_array(
            self.textured, [(self.quad_buffer, "2f 2f", "position", "uv")]
        )
        self.wood_quad = self.ctx.vertex_array(self.wood, [(self.quad_buffer, "2f 8x", "position")])
        self.textures = {}
        self.labels = OrderedDict()
        self.font = pygame.font.SysFont("arial", 14)
        self.hud = None
        self.parts = []
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def triangles(self, points, color):
        points = np.asarray(points, dtype="f4").reshape(-1, 2)
        data = np.empty((len(points), 6), dtype="f4")
        data[:, :2] = points
        data[:, 2:] = np.array((*color[:3], color[3] if len(color) > 3 else 255)) / 255.0
        self.parts.append(data)

    def strip(self, left, right, color):
        left, right = np.asarray(left), np.asarray(right)
        self.triangles(
            np.stack((left[:-1], right[:-1], left[1:], right[:-1], right[1:], left[1:]), axis=1),
            color,
        )

    def segments(self, a, b, color, width=1):
        a, b = np.asarray(a), np.asarray(b)
        if not len(a):
            return
        delta = b - a
        lengths = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
        n = np.column_stack((-delta[:, 1], delta[:, 0])) / lengths[:, None] * width / 2
        self.triangles(np.stack((a + n, a - n, b + n, a - n, b - n, b + n), axis=1), color)

    def lines(self, points, color, width=1):
        points = np.asarray(points)
        if len(points) > 1:
            self.segments(points[:-1], points[1:], color, width)

    def circle(self, center, radius, color):
        angles = np.linspace(0, 2 * np.pi, 41)
        self.lines(
            np.asarray(center) + radius * np.column_stack((np.cos(angles), np.sin(angles))),
            color,
            1.2,
        )

    def flush(self):
        if self.parts:
            data = np.concatenate(self.parts)
            if data.nbytes > self.buffer.size:
                self.buffer.orphan(data.nbytes)
            self.buffer.write(data)
            self.mesh.render(moderngl.TRIANGLES, vertices=len(data))
            self.parts.clear()

    def texture(self, surface):
        native = surface.get_pitch() == surface.get_width() * 4 and surface.get_masks() == (
            16711680,
            65280,
            255,
            4278190080,
        )
        data = surface.get_view("1") if native else pygame.image.tobytes(surface, "RGBA")
        result = self.ctx.texture(surface.get_size(), 4, data)
        result.swizzle = "BGRA" if native else "RGBA"
        result.filter = (moderngl.LINEAR, moderngl.LINEAR)
        result.repeat_x = result.repeat_y = False
        return result

    def blit(self, texture, center, size, angle=0):
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]) * size
        c, s = np.cos(angle), np.sin(angle)
        corners = corners @ np.array([[c, s], [-s, c]]) + center
        data = np.column_stack((corners, [[0, 0], [1, 0], [0, 1], [1, 1]])).astype("f4")
        self.quad_buffer.write(data)
        texture.use()
        self.quad.render(moderngl.TRIANGLE_STRIP)

    def label(self, text, position, color):
        key = (text, tuple(color))
        if key not in self.labels:
            self.labels[key] = self.texture(self.font.render(text, True, color))
            if len(self.labels) > 64:
                self.labels.popitem(last=False)[1].release()
        self.labels.move_to_end(key)
        texture = self.labels[key]
        self.blit(texture, np.asarray(position) + np.array(texture.size) / 2, texture.size)

    def draw(self, frame, ui, trails, colors, size, patches, rain_field, impacts):
        width, height = size
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, width, height)
        for program in (self.solid, self.textured, self.wood):
            program["size"].value = size
        states = np.asarray(frame["states"])
        follow = min(ui["follow"], len(states) - 1)
        center = np.asarray(frame["center"])
        origin = states[follow, :2]
        yaw = ui["camera_yaw"]
        c, s = np.cos(yaw), np.sin(yaw)
        scale = min(width / 40, height / 32)
        anchor = np.array([width * 0.53, height * 0.64])
        if ui["overview"]:
            factor = min(1.0, width / 1280.0, height / 900.0)
            margin = (310 if ui.get("infos", True) else 32) * factor
            lo, hi = center[:, :2].min(axis=0) - 6, center[:, :2].max(axis=0) + 6
            origin = (lo + hi) / 2
            scale = min(
                (width - 2 * margin) / max(1.0, hi[0] - lo[0]),
                (height - 235 * factor) / max(1.0, hi[1] - lo[1]),
            )
            anchor = np.array([width / 2, (height + 5 * factor) / 2])

        def project(points):
            p = np.asarray(points) - origin
            if ui["overview"]:
                return p * [scale, -scale] + anchor
            return (
                np.column_stack((-s * p[:, 0] + c * p[:, 1], -c * p[:, 0] - s * p[:, 1])) * scale
                + anchor
            )

        self.wood["origin"].value = tuple(origin)
        self.wood["anchor"].value = tuple(anchor)
        self.wood["heading"].value = (c, s)
        self.wood["scale"].value = scale
        self.wood["overview"].value = ui["overview"]
        self.quad_buffer.write(
            np.array(
                [[0, 0, 0, 0], [width, 0, 1, 0], [0, height, 0, 1], [width, height, 1, 1]],
                dtype="f4",
            )
        )
        self.wood_quad.render(moderngl.TRIANGLE_STRIP)
        normals = np.column_stack((-np.sin(center[:, 2]), np.cos(center[:, 2])))
        road = center[:, :2]
        half = frame["width"] / 2
        left, right = project(road + normals * half), project(road - normals * half)
        outer_l, outer_r = (
            project(road + normals * (half + 0.38)),
            project(road - normals * (half + 0.38)),
        )
        self.strip(outer_l + [5, 7], outer_r + [5, 7], (62, 39, 30))
        self.strip(outer_l, outer_r, (128, 153, 151))
        rain = frame["rain"]
        self.strip(left, right, (56, 65 + int(7 * rain), 73 + int(16 * rain)))
        self.lines(left, (231, 229, 204), max(2, scale * 0.15))
        self.lines(right, (231, 229, 204), max(2, scale * 0.15))
        if patches:
            points = project(np.asarray(list(patches.values())).reshape(-1, 2)).reshape(-1, 8, 2)
            indices = np.array([[0, i, i + 1] for i in range(1, 7)])
            self.triangles(points[:, indices], (49, 67, 82))
        cp = project(road)
        marked = (np.floor(center[:-1, 3] / 2).astype(int) % 2) == 0
        self.segments(cp[:-1][marked], cp[1:][marked], (102, 120, 126))
        self.segments(left[:-1][marked], left[1:][marked], (232, 83, 57), max(3, scale * 0.19))
        self.segments(right[:-1][marked], right[1:][marked], (232, 83, 57), max(3, scale * 0.19))
        sectors = np.floor(center[:, 3] / frame["sector_length"]).astype(int)
        marked = sectors[:-1] != sectors[1:]
        self.segments(left[:-1][marked], right[:-1][marked], (232, 206, 129), 3)
        sector_labels = [
            (f"S{sectors[i + 1]}", cp[i] + [10, 0])
            for i in np.flatnonzero(marked)
            if 0 <= cp[i, 0] < width and 0 <= cp[i, 1] < height
        ]
        self.lines([left[0], right[0]], (222, 99, 114), 4)
        self.lines([left[-1], right[-1]], (247, 192, 88), 4)
        for i, trail in enumerate(trails):
            if ui.get("trails", True) and len(trail) > 1:
                self.lines(project(trail), tuple(int(x * 0.6) for x in colors[i]), 2)
        if ui["lidar"]:
            from crashlearn_sim.simulation.physics import LIDAR_ANGLES

            angles = LIDAR_ANGLES + states[follow, 4]
            pos = states[follow, :2]
            ends = pos + np.asarray(frame["lidar"][follow])[:, None] * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
            starts = np.repeat(project([pos]), len(ends), axis=0)
            ends = project(ends)
            near = np.asarray(frame["lidar"][follow]) <= 0.11
            self.segments(starts[~near], ends[~near], (52, 116, 137))
            self.segments(starts[near], ends[near], (249, 115, 102))
        self.flush()
        for i, car in enumerate(states):
            if frame["status"][i] != 1:
                continue
            dot = project([car[:2]])[0]
            angle = np.pi / 2 - car[4] if ui["overview"] else car[4] - yaw
            length = max(24.0, min(78.0, scale * 1.5))
            if i == follow:
                self.circle(dot, length * 0.57, colors[i])
            if rain > 0.2:
                tail = project([car[:2] - 0.7 * np.array([np.cos(car[4]), np.sin(car[4])])])[0]
                self.circle(tail, 3 + rain * 3, (109, 149, 169))
            if i in impacts or frame["wall_hits"][i] or frame["vehicle_hits"][i]:
                age = impacts.get(i, 0.0)
                self.circle(dot, 18 + age * 13, (255, 144, 80, int(255 * (1 - age))))
                angles = np.arange(8) * np.pi / 4 + frame["time"] * 0.5
                rays = np.column_stack((np.cos(angles), np.sin(angles)))
                self.segments(
                    dot + rays * (17 + age * 13),
                    dot + rays * (29 + age * 23),
                    (255, 204, 105, int(255 * (1 - age))),
                    2.5,
                )
                self.segments(
                    dot + rays * (18 + age * 13),
                    dot + rays * (23 + age * 18),
                    (255, 248, 211, int(255 * (1 - age))),
                    1.3,
                )
            self.flush()
            color = tuple(colors[i])
            if color not in self.textures:
                self.textures[color] = self.texture(car_model(color))
                self.textures[color].build_mipmaps()
                self.textures[color].filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.blit(self.textures[color], dot, (length * 160 / 260, length), angle)
            self.label(str(i + 1), dot + [length * 0.4, -8], color)
        for text, position in sector_labels:
            self.label(text, position, (232, 206, 129))
        if rain > 0.05:
            self.triangles(
                [[0, 0], [width, 0], [0, height], [width, 0], [width, height], [0, height]],
                (64, 85, 111, int(25 * rain)),
            )
            starts, ends, brightness = rain_field.segments(frame["time"], size, rain)
            self.segments(starts, ends, (110, 158, 189))
            if rain > 0.7 and frame["time"] % 13 < 0.1:
                self.triangles(
                    [[0, 0], [width, 0], [0, height], [width, 0], [width, height], [0, height]],
                    (185, 210, 231, 24),
                )
            self.flush()

    def overlay(self, surface, update=True):
        if self.hud is None or self.hud.size != surface.get_size():
            if self.hud is not None:
                self.hud.release()
            self.hud = self.texture(surface)
        elif update:
            self.hud.write(surface.get_view("1"))
        self.blit(self.hud, np.array(surface.get_size()) / 2, surface.get_size())

    def screenshot(self, path):
        size = pygame.display.get_window_size()
        image = pygame.image.frombytes(
            self.ctx.screen.read(viewport=(0, 0, *size), components=3), size, "RGB"
        )
        pygame.image.save(pygame.transform.flip(image, False, True), str(path))

    def release(self):
        for item in (
            *self.textures.values(),
            *self.labels.values(),
            self.hud,
            self.mesh,
            self.quad,
            self.wood_quad,
            self.buffer,
            self.quad_buffer,
            self.solid,
            self.textured,
            self.wood,
        ):
            if item is not None:
                item.release()
        self.ctx.release()

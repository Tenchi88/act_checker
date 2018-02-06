import numpy as np
from PIL import Image, ImageDraw, ImageFont
from math import atan2, pi
import csv


class PageChecker:
    def __init__(self, input_image):
        self.image = Image.open(input_image)
        self.original_x, self.original_y = self.image.size
        self.resized_im = None
        self.image_data = None
        self.horizontal_raw = []
        self.vertical_raw = []
        self.image_with_lines = None
        self.labels = []
        self.output_image = None
        self.draw = None
        self.areas = []
        self.x_size = 0
        self.y_size = 0
        self.horizontal = []
        self.vertical = []

        self.zones = {
            'Зона 1': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 2': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 3': {
                'thr': 240,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 4': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 5': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 6': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 7': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 8': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 9': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 10': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 11': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 12': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 13': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 14': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 15': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 16': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 17': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 18': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 19': {
                'thr': 230,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 20': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 21a': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 21b': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 22': {
                'thr': 240,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 23': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 24': {
                'thr': 240,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 25': {
                'thr': 220,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 26': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 27': {
                'thr': 253,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 28': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 29a': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 29b': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 30': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 31a': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 31b': {
                'thr': 245,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 32': {
                'thr': None,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 33': {
                'thr': 249,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 34a': {
                'thr': 240,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 34b': {
                'thr': 240,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 35a': {
                'thr': 240,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
            'Зона 35b': {
                'thr': 240,
                'ok': 'Заполнена',
                'error': 'Не заполнена',
            },
        }

    def resize_to_array(self, verbose=False):
        scale = 2
        self.x_size = self.original_x // scale
        self.y_size = self.original_y // scale
        self.resized_im = self.image.resize((self.x_size, self.y_size), Image.ANTIALIAS)
        scale = 4
        self.x_size = self.original_x // scale
        self.y_size = self.original_y // scale
        self.resized_im = self.resized_im.resize((self.x_size, self.y_size), Image.ANTIALIAS)
        self.image_data = np.asarray(self.resized_im).copy()
        if verbose:
            print('Размер сжатого изображения: {} x {}'.format(self.x_size, self.y_size))

    def search_lines(
            self,
            line_len_horizontal=10, line_intense_horizontal=150,
            line_len_vertical=10, line_intense_vertical=170,
            lines_only=False
    ):
        if lines_only:
            new_data = np.zeros([len(self.image_data), len(self.image_data[0]), 3], dtype=np.ubyte)
            new_data.fill(255)
        else:
            new_data = np.copy(self.image_data)
        maybe_line = False
        for x in range(len(self.image_data)):
            start = 0
            cnt = 0
            for y in range(len(self.image_data[0])):
                if self.image_data[x][y][0] < line_intense_horizontal:
                    if maybe_line:
                        cnt += 1
                    else:
                        maybe_line = True
                        start = y
                        cnt = 1
                else:
                    if maybe_line:
                        maybe_line = False
                        if cnt > line_len_horizontal:
                            line = {
                                'level': x,
                                'start': start,
                                'len': cnt
                            }
                            self.horizontal_raw.append(line)

        maybe_line = False
        for y in range(len(self.image_data[0])):
            start = 0
            cnt = 0
            for x in range(len(self.image_data)):
                if self.image_data[x][y][0] < line_intense_vertical:
                    if maybe_line:
                        cnt += 1
                    else:
                        maybe_line = True
                        start = x
                        cnt = 1
                else:
                    if maybe_line:
                        maybe_line = False
                        if cnt > line_len_vertical:
                            line = {
                                'level': y,
                                'start': start,
                                'len': cnt
                            }
                            self.vertical_raw.append(line)
        self.image_with_lines = new_data

    def draw_horizontal_line(self, line, color=(255, 0, 0, 255), width=1):
        self.draw.line(
            [(line['start']), line['level'], (line['start'] + line['len']), line['level']],
            fill=color,
            width=width,
        )

    def draw_vertical_line(self, line, color=(0, 255, 0, 255), width=1):
        self.draw.line(
            [(line['level'], line['start']), (line['level'], line['start'] + line['len'])],
            fill=color,
            width=width,
        )

    def draw_area(self, x, y, x_size, y_size, color=(0, 255, 255, 255), width=1):
        self.draw.line(
            [(x, y), (x, y + y_size)], fill=color, width=width
        )
        self.draw.line(
            [(x, y), (x + x_size, y)], fill=color, width=width
        )
        self.draw.line(
            [(x + x_size, y), (x + x_size, y + y_size)], fill=color, width=width
        )
        self.draw.line(
            [(x, y + y_size), (x + x_size, y + y_size)], fill=color, width=width
        )

    def add_area(
            self, x, y, x_size, y_size, color=(0, 255, 255, 255), width=1,
            label=None, label_font_size=14, label_color=(0, 0, 255, 255)
    ):
        self.areas.append(
            {
                'x': int(x),
                'y': int(y),
                'x_size': int(x_size),
                'y_size': int(y_size),
                'color': color,
                'width': width,
                'label': label,
            }
        )
        if label:
            self.add_label((x + x_size + 4, y), label, font_size=label_font_size, color=label_color)

    def draw_areas(self):
        for area in self.areas:
            self.draw_area(area['x'], area['y'], area['x_size'], area['y_size'], area['color'], area['width'])

    def draw_lines(self):
        for line in self.horizontal:
            self.draw_horizontal_line(line)
        for line in self.vertical:
            self.draw_vertical_line(line)

    def draw_raw_lines(self):
        for line in self.horizontal_raw:
            self.draw_horizontal_line(line)
        for line in self.vertical_raw:
            self.draw_vertical_line(line)

    def add_label(self, xy, text, font_size=14, color=(0, 0, 255, 255)):
        self.labels.append(
            {
                'xy': xy,
                'text': text,
                'font': ImageFont.truetype('fonts/OpenSans-Regular.ttf', font_size),
                'color': color
            }
        )

    def draw_all_marks(self):
        self.draw_lines()
        self.draw_labels()
        self.draw_areas()

    def convert_to_image(self):
        self.output_image = Image.fromarray(self.image_with_lines)
        self.draw = ImageDraw.Draw(self.output_image)

    def draw_labels(self):
        for label in self.labels:
            self.draw.text(label['xy'], label['text'], font=label['font'], fill=label['color'])

    def save_to(self, file_name):
        self.output_image.save(file_name)

    @staticmethod
    def lines_in_area(lines, min_level, max_level, search_param='level', with_borders=False):
        a, b = None, None
        for i in range(len(lines)):
            if with_borders:
                if lines[i][search_param] >= min_level:
                    if a is None:
                        a = i
                    b = i
                if lines[i][search_param] > max_level:
                    break
            else:
                if lines[i][search_param] > min_level:
                    if a is None:
                        a = i
                    b = i
                if lines[i][search_param] >= max_level:
                    break
        return lines[a:b + 1]

    @staticmethod
    def _lines_is_crossed(a, b, delta):
        if a['start'] <= b['start'] + delta and a['start'] + a['len'] + delta >= b['start']:
            return True
        if a['start'] <= b['start'] + b['len'] + delta and a['start'] + a['len'] + delta >= b['start'] + b['len']:
            return True
        if a['start'] + delta >= b['start'] and a['start'] + a['len'] <= b['start'] + b['len'] + delta:
            return True
        if b['start'] + delta >= a['start'] and b['start'] + b['len'] <= a['start'] + a['len'] + delta:
            return True
        return False
    
    def _join_lines_from_list(self, input_list, delta_along, delta_cross):
        joined = []
        current_line = input_list[0].copy()
        tmp_list = input_list.copy()
        tmp_list.remove(current_line)
        while len(tmp_list):
            used_lines = []
            for next_line in tmp_list:
                if (
                        current_line['level'] + delta_cross >= next_line['level'] and
                        self._lines_is_crossed(current_line, next_line, delta_along)
                ):
                    end = max(
                        current_line['start'] + current_line['len'],
                        next_line['start'] + next_line['len']
                    )
                    current_line['start'] = min(
                        current_line['start'],
                        next_line['start']
                    )
                    current_line['len'] = end - current_line['start']
                    current_line['level'] = next_line['level']
                    used_lines.append(next_line)
            for line in used_lines:
                tmp_list.remove(line)
                if len(tmp_list) == 0:
                    joined.append(current_line)
            if len(used_lines) == 0:
                joined.append(current_line)
                current_line = tmp_list[0].copy()
                tmp_list.remove(tmp_list[0])
                if len(tmp_list) == 0:
                    joined.append(current_line)
        # for j in range(1, len(input_list)):
        #     for i in range(j, len(input_list)):
        #         if (
        #                 current_line['level'] + delta_cross >= input_list[i]['level'] and
        #                 self._lines_is_crossed(current_line, input_list[i], delta_along)
        #         ):
        #             end = max(
        #                 current_line['start'] + current_line['len'],
        #                 input_list[i]['start'] + input_list[i]['len']
        #             )
        #             current_line['start'] = min(
        #                 current_line['start'],
        #                 input_list[i]['start']
        #             )
        #             current_line['len'] = end - current_line['start']
        #             current_line['level'] = input_list[i]['level']
        #         elif i == len(input_list)-1:
        #             joined.append(current_line)
        #             current_line = input_list[j].copy()
        return joined

    def join_lines(self, delta_along=2, delta_cross=2):
        self.horizontal_raw = self._join_lines_from_list(self.horizontal_raw, delta_along, delta_cross)
        self.vertical_raw = self._join_lines_from_list(self.vertical_raw, delta_along, delta_cross)

    @staticmethod
    def _search_lines_with_close_endings(lines, count, tolerance=0.1, verbose=False):
        tmp_list = lines.copy()
        result = []
        while len(tmp_list) >= count:
            current_list = [tmp_list[0].copy()]
            tmp_list.remove(tmp_list[0])
            val = current_list[0]['start'] + current_list[0]['len']
            used_lines = []
            for line in tmp_list:
                if line['len'] < 20:
                    continue
                if line['start'] + line['len']*(1.0-tolerance) <= val <= line['start'] + line['len']*(1.0+tolerance):
                    used_lines.append(line)
            for line in used_lines:
                current_list.append(line.copy())
                tmp_list.remove(line)
            if len(current_list) > count:
                if verbose:
                    print('Найден набор линий длинной', len(current_list))
                result.append(current_list.copy())
        return result

    def search_in_page_1(self, verbose=False):
        # Находим зону 4 "УИК,ТИК №"
        line_size = 15
        area_4_line = None
        area_3_line = None
        for line in self.horizontal_raw:
            if line['level'] > self.y_size * 0.1 and line['len'] > self.x_size * 0.2:
                print('Линия для зоны 4(Номер УИК)', line)
                area_4_line = line
                # self.horizontal.append(area_4_line)
                break

        for i in range(2, len(self.horizontal_raw)):
            area_3_line = self.horizontal_raw[i-1]
            if self.horizontal_raw[i] == area_4_line:
                break
        area_3_line['start'] = area_4_line['start'] + area_4_line['len'] + 20
        area_3_line['len'] = area_4_line['len'] - 30
        # self.horizontal.append(area_3_line)
        print('Линия для зоны 3(Дата)', area_3_line)

        # Находим зону 13 "Таблица выездов"
        vertical_groups = self._search_lines_with_close_endings(self.vertical_raw, 5, verbose=verbose)
        group_areas = []
        for group in vertical_groups:
            low_x = self.x_size
            low_y = self.y_size
            high_x = 0
            high_y = 0
            for line in group:
                self.vertical.append(line)
                low_x = min(low_x, line['level'])
                high_x = max(high_x, line['level'])
                low_y = min(low_y, line['start'])
                high_y = max(high_y, line['start'] + line['len'])
            group_areas.append(
                {
                    'x': low_x,
                    'y': low_y,
                    'x_size': high_x-low_x,
                    'y_size': high_y-low_y,
                }
            )
        area_labels = ['Зона 13', 'Зона 16']
        table_number = 0
        for area in group_areas:
            self.add_area(
                area['x'], area['y'], area['x_size'], area['y_size'],
                label=area_labels[table_number],
                label_font_size=10
            )
            table_number += 1

        zone_14_start = min(group_areas[0]['y'], group_areas[1]['y'])
        address_lines = []
        additive = 10
        for line in self.horizontal_raw:
            if area_4_line['level'] + additive < line['level'] < zone_14_start - additive:
                address_lines.append(line.copy())

        if len(address_lines) > 1:
            low_x = self.x_size
            low_y = self.y_size
            high_x = 0
            high_y = 0
            for line in address_lines:
                self.horizontal.append(line)
                low_x = min(low_x, line['start'])
                high_x = max(high_x, line['start'] + line['len'])
                low_y = min(low_y, line['level'])
                high_y = max(high_y, line['level'])
            addr_group = {
                    'x': low_x,
                    'y': low_y-20,
                    'x_size': high_x-low_x,
                    'y_size': high_y-low_y+30,
            }
            self.add_area(
                addr_group['x'], addr_group['y'], addr_group['x_size'], addr_group['y_size'],
                label='Адрес',
                label_font_size=10
            )
            # line_size = (address_lines[2]['level']-address_lines[0]['level'])/2
            self.add_area(
                area_4_line['start'], area_4_line['level']-line_size, area_4_line['len'], line_size,
                label='Зона 4',
                label_font_size=10
            )
            self.add_area(
                area_3_line['start'], area_3_line['level']-line_size, area_3_line['len'], line_size,
                label='Зона 3',
                label_font_size=10
            )

            zone_number = 5
            prev_level, next_level = None, None
            for line in address_lines:
                prev_level = next_level
                next_level = line['level']
                if prev_level is not None and next_level is not None:
                    if next_level - prev_level > 1.8 * line_size:
                        zone_number += 1
                self.add_area(
                    line['start'], line['level']-line_size, line['len'], line_size,
                    label='Зона {}'.format(zone_number),
                    label_font_size=10
                )
                zone_number += 1

        zone_16_end = max(group_areas[0]['y']+group_areas[0]['y_size'], group_areas[1]['y']+group_areas[1]['y_size'])
        lines_after_16 = []
        additive = 10
        for line in self.horizontal_raw:
            if zone_16_end + additive < line['level'] and line['len'] > 100:
                lines_after_16.append(line.copy())
        zone_number = 17
        for line in lines_after_16:
            # self.horizontal.append(line)
            print(line)
            self.add_area(
                line['start'], line['level']-line_size, line['len'], line_size,
                label='Зона {}'.format(zone_number),
                label_font_size=10
            )
            zone_number += 1

    def mean_val_in_area(self, verbose=False):
        for area in self.areas:
            area_data = self.image_data[area['y']+1:area['y']+area['y_size']-1, area['x']+1:area['x']+area['x_size']-1]
            val = area_data.mean()
            if verbose:
                print('Среднее значение в зоне', area['label'], val)
            if area['label'] in self.zones:
                self.zones[area['label']]['val'] = val

    def color_results(self):
        for zone in self.zones:
            if self.zones[zone]['thr'] is None:
                pass
            elif 'val' not in self.zones[zone]:
                pass
            elif self.zones[zone]['val'] <= self.zones[zone]['thr']:
                pass
            else:
                for area in self.areas:
                    if area['label'] == zone:
                        area['color'] = (255, 0, 0, 255)

    def show_results(self):
        for zone in self.zones:
            if self.zones[zone]['thr'] is None:
                print('{} не поддерживается'.format(zone))
            elif 'val' not in self.zones[zone]:
                print('{} не найдена'.format(zone))
            elif self.zones[zone]['val'] <= self.zones[zone]['thr']:
                print('{}: {} [{} < {}]'.format(
                    zone, self.zones[zone]['ok'], int(self.zones[zone]['val']), self.zones[zone]['thr'])
                )
            else:
                print('{}: {} [{} < {}]'.format(
                    zone, self.zones[zone]['error'], int(self.zones[zone]['val']), self.zones[zone]['thr'])
                )

    def detect_rotation(self, thr=3, verbose=False, group_size=4):
        order = None
        group = []
        for i in range(1, len(self.horizontal_raw)):
            if self.horizontal_raw[i]['level'] - self.horizontal_raw[i-1]['level'] <= thr:
                if order == 'forward':
                    if (
                        self.horizontal_raw[i]['start'] >=
                        self.horizontal_raw[i - 1]['start'] + self.horizontal_raw[i-1]['len']
                    ):
                        group.append(self.horizontal_raw[i])
                elif order == 'backward':
                    if (
                        self.horizontal_raw[i]['start'] + self.horizontal_raw[i]['len'] <=
                        self.horizontal_raw[i-1]['start']
                    ):
                        group.append(self.horizontal_raw[i])
                else:
                    if (
                        self.horizontal_raw[i]['start'] + self.horizontal_raw[i]['len'] <=
                        self.horizontal_raw[i-1]['start']
                    ):
                        order = 'backward'
                        group.append(self.horizontal_raw[i-1])
                        group.append(self.horizontal_raw[i])
                    elif (
                        self.horizontal_raw[i]['start'] >=
                        self.horizontal_raw[i - 1]['start'] + self.horizontal_raw[i-1]['len']
                    ):
                        order = 'forward'
                        group.append(self.horizontal_raw[i-1])
                        group.append(self.horizontal_raw[i])
            else:
                if len(group) > 4:
                    total_len = 0
                    for line in group:
                        total_len += line['len']
                    if total_len > self.x_size*2/3:
                        if order == 'forward':
                            dx = group[-1]['start'] + group[-1]['len'] - group[0]['start']
                            dy = group[-1]['level'] - group[0]['level']
                        else:
                            dx = group[0]['start'] + group[0]['len'] - group[-1]['start']
                            dy = group[-1]['level'] - group[0]['level']
                        angle = atan2(dy, dx)*180.0/pi
                        if verbose:
                            print('Угол: ', angle)
                        if -0.5 <= angle <= 0.5:
                            return
                        self.image = self.image.rotate(angle, resample=Image.NEAREST)
                        self.image_prepare(delta_along=5, line_intense_horizontal=190, line_intense_vertical=190)
                        break
                order = None
                group = []

    def image_prepare(self, delta_along=3, delta_cross=2, line_intense_horizontal=150, line_intense_vertical=170):
        self.original_x, self.original_y = self.image.size
        self.resized_im = None
        self.image_data = None
        self.horizontal_raw = []
        self.vertical_raw = []
        self.image_with_lines = None
        self.labels = []
        self.output_image = None
        self.draw = None
        self.areas = []
        self.x_size = 0
        self.y_size = 0
        self.horizontal = []
        self.vertical = []

        self.resize_to_array(verbose=True)
        self.search_lines(
            lines_only=False,
            line_intense_horizontal=line_intense_horizontal,
            line_intense_vertical=line_intense_vertical
        )
        self.convert_to_image()
        self.join_lines(delta_along=delta_along, delta_cross=delta_cross)

    def search_in_page_2(self, verbose=False):
        # fixme
        line_size = 15
        area_20_line = None
        for line in self.horizontal_raw:
            if line['len'] > 50 and line['level'] > 0.05*self.y_size:
                area_20_line = line
                self.add_area(
                    line['start'], line['level']-line_size, line['len'], line_size,
                    label='Зона 20', label_font_size=10
                )
                break
        large_field_lines = []
        for line in self.horizontal_raw:
            if line['len'] > 0.7*self.x_size and line['level'] > area_20_line['level']:
                large_field_lines.append(line)
        lines_in_area_a = []
        area_21_line_1 = None
        area_21_line_2 = None
        if len(large_field_lines) > 2:
            area_a_x = min(large_field_lines[0]['start'], large_field_lines[1]['start'])
            area_a_x_size = max(large_field_lines[0]['len'], large_field_lines[1]['len'])
            area_a_y = large_field_lines[0]['level']
            area_a_y_size = large_field_lines[1]['level'] - large_field_lines[0]['level']
            for line in self.horizontal_raw:
                if (area_a_y + 10 < line['level'] < area_a_y + area_a_y_size - 10) and line['len'] > 30:
                    lines_in_area_a.append(line)
            sum_line = self._join_lines_from_list(lines_in_area_a, delta_along=10, delta_cross=3)
            for line in sum_line:
                if line['len'] > 0.5*area_a_x_size:
                    if area_21_line_1 is None:
                        area_21_line_1 = line
                    else:
                        area_21_line_2 = line
                        break
            if area_21_line_1:
                self.add_area(
                    area_21_line_1['start'], area_21_line_1['level']-line_size, area_21_line_1['len'], line_size,
                    label='Зона 21a', label_font_size=10
                )
            if area_21_line_2:
                self.add_area(
                    area_21_line_2['start'], area_21_line_2['level']-line_size, area_21_line_2['len'], line_size,
                    label='Зона 21b', label_font_size=10
                )
            area_b_x = min(large_field_lines[1]['start'], large_field_lines[2]['start'])
            area_b_x_size = max(large_field_lines[1]['len'], large_field_lines[2]['len'])
            area_b_y = large_field_lines[1]['level']
            area_b_y_size = large_field_lines[2]['level'] - large_field_lines[1]['level']
            # self.add_area(
            #     area_b_x, area_b_y, area_b_x_size, area_b_y_size,
            #     label='Зона 22', label_font_size=10
            # )
            self.add_area(
                area_b_x+0.08*area_b_x_size, area_b_y,
                area_b_x_size/2-0.15*area_b_x_size, area_b_y_size/2,
                label='Зона 22', label_font_size=10
            )
            self.add_area(
                area_b_x+area_b_x_size/2, area_b_y,
                area_b_x_size/2-0.1*area_b_x_size, area_b_y_size/2,
                label='Зона 223', label_font_size=10
            )
            self.add_area(
                area_b_x+0.08*area_b_x_size, area_b_y+area_b_y_size/2,
                area_b_x_size/2-0.15*area_b_x_size, area_b_y_size/2,
                label='Зона 224', label_font_size=10
            )

    def search_in_page_3(self, verbose=False):
        line_size = 15
        area_25_line = None
        for line in self.horizontal_raw:
            if line['level'] > 0.1*self.y_size and line['len'] > 20:
                area_25_line = line
                break
        if area_25_line:
            self.add_area(
                area_25_line['start'], area_25_line['level']-line_size, area_25_line['len'], line_size,
                label='Зона 25', label_font_size=10
            )
        longest_horizontal = None
        longest_vertical = None
        for line in self.horizontal_raw:
            if longest_horizontal is None or longest_horizontal['len'] < line['len']:
                longest_horizontal = line
        for line in self.vertical_raw:
            if longest_vertical is None or longest_vertical['len'] < line['len']:
                longest_vertical = line
        area_x = min(longest_horizontal['start'], longest_vertical['level'])
        area_x_size = longest_horizontal['len']
        area_y = min(longest_horizontal['level'], longest_vertical['start'])
        area_y_size = longest_vertical['len']
        self.add_area(area_x, area_y, area_x_size, area_y_size, label='Зона 27')

        sign_lines = []
        for line in self.horizontal_raw:
            if line['level'] > area_y + area_y_size + 20 and line['len'] > 50:
                sign_lines.append(line)
        area_28_line = None
        area_29a_line = None
        area_29b_line = None
        for line in sign_lines:
            if line['start'] > self.x_size/2 - 30:
                if area_29a_line is None:
                    area_29a_line = line
                elif area_29b_line is None:
                    area_29b_line = line
            elif area_28_line is None:
                area_28_line = line

        if area_28_line:
            self.add_area(
                area_28_line['start'], area_28_line['level']-line_size, area_28_line['len'], line_size,
                label='Зона 28', label_font_size=10
            )
        if area_29a_line:
            self.add_area(
                area_29a_line['start'], area_29a_line['level']-line_size, area_29a_line['len'], line_size,
                label='Зона 29a', label_font_size=10
            )
        if area_29b_line:
            self.add_area(
                area_29b_line['start'], area_29b_line['level']-line_size, area_29b_line['len'], line_size,
                label='Зона 29b', label_font_size=10
            )

        level = max((area_28_line['level'], area_29a_line['level'], area_29b_line['level']))
        area_30_line = None
        for line in self.horizontal_raw:
            if line['level'] > level and line['len'] > 70:
                area_30_line = line
        if area_30_line:
            self.add_area(
                area_30_line['start'], area_30_line['level']-line_size, area_30_line['len'], line_size,
                label='Зона 30', label_font_size=10
            )
        level = max(level, area_30_line['level'])
        area_31a_line = None
        area_31b_line = None
        lines_31 = []
        for line in self.horizontal_raw:
            if line['level'] > level + 10 and line['len'] > 30:
                lines_31.append(line)
                if len(lines_31) == 2:
                    if lines_31[0]['start'] < lines_31[1]['start']:
                        area_31a_line = lines_31[0]
                        area_31b_line = lines_31[1]
                    else:
                        area_31a_line = lines_31[1]
                        area_31b_line = lines_31[0]
        if area_31a_line:
            self.add_area(
                area_31a_line['start'], area_31a_line['level']-line_size, area_31a_line['len'], line_size,
                label='Зона 31a', label_font_size=10
            )
        if area_31b_line:
            self.add_area(
                area_31b_line['start'], area_31b_line['level']-line_size, area_31b_line['len'], line_size,
                label='Зона 31b', label_font_size=10
            )

    def search_in_page_6(self, verbose=False):
        line_size = 15
        longest_horizontal = None
        longest_vertical = None
        for line in self.horizontal_raw:
            if longest_horizontal is None or longest_horizontal['len'] < line['len']:
                longest_horizontal = line
        for line in self.vertical_raw:
            if longest_vertical is None or longest_vertical['len'] < line['len']:
                longest_vertical = line
        area_x = min(longest_horizontal['start'], longest_vertical['level'])
        area_x_size = longest_horizontal['len']
        area_y = min(longest_horizontal['level'], longest_vertical['start'])
        area_y_size = longest_vertical['len']
        self.add_area(area_x, area_y, area_x_size, area_y_size, label='Зона 32', label_font_size=8)

        area_33_line = None
        area_34a_line = None
        area_34b_line = None
        sign_lines = []
        for line in self.horizontal_raw:
            if line['level'] > area_y + area_y_size + 20 and line['len'] > 50:
                sign_lines.append(line)
        for line in sign_lines:
            if line['start'] > self.x_size/2 - 30:
                if area_34a_line is None:
                    area_34a_line = line
                elif area_34b_line is None:
                    area_34b_line = line
            elif area_33_line is None:
                area_33_line = line

        if area_33_line:
            self.add_area(
                area_33_line['start'], area_33_line['level']-line_size, area_33_line['len'], line_size,
                label='Зона 33', label_font_size=10
            )
        if area_34a_line:
            self.add_area(
                area_34a_line['start'], area_34a_line['level']-line_size, area_34a_line['len'], line_size,
                label='Зона 34a', label_font_size=10
            )
        if area_34b_line:
            self.add_area(
                area_34b_line['start'], area_34b_line['level']-line_size, area_34b_line['len'], line_size,
                label='Зона 34b', label_font_size=10
            )

        level = max((area_33_line['level'], area_34a_line['level'], area_34b_line['level']))
        area_35a_line = None
        area_35b_line = None
        for line in self.horizontal_raw:
            if line['level'] > level + 10 and line['len'] > 30:
                if area_35a_line is None:
                    area_35a_line = line
                elif area_35b_line is None:
                    area_35b_line = line
        if area_35a_line:
            self.add_area(
                area_35a_line['start'], area_35a_line['level']-line_size, area_35a_line['len'], line_size,
                label='Зона 35a', label_font_size=10
            )
        if area_35b_line:
            self.add_area(
                area_35b_line['start'], area_35b_line['level']-line_size, area_35b_line['len'], line_size,
                label='Зона 35b', label_font_size=10
            )


class Checker:
    def __init__(self, name, pages=1):
        self.name = name
        self.page = []
        for i in range(1, pages+1):
            self.page.append(PageChecker('{}-{}.jpg'.format(name, i)))
        self.results = {}

    def check_page_1(self):
        page = self.page[0]
        print('Размер исходного  изображения:', page.original_x, page.original_y)
        page.resize_to_array(verbose=True)
        page.search_lines(lines_only=False)
        page.convert_to_image()
        page.join_lines(delta_along=3, delta_cross=2)
        # page.detect_rotation()
        page.search_in_page_1(verbose=True)
        page.mean_val_in_area()
        # checker.draw_raw_lines()
        page.color_results()
        page.draw_all_marks()
        # page.show_results()
        page.save_to('{}-{}_edited.jpg'.format(self.name, 1))

    def check_page_2(self):
        page = self.page[1]
        print('Размер исходного  изображения:', page.original_x, page.original_y)
        page.resize_to_array(verbose=True)
        page.search_lines(lines_only=False)
        page.convert_to_image()
        page.join_lines(delta_along=3, delta_cross=2)
        page.detect_rotation()
        page.search_in_page_2(verbose=True)
        page.mean_val_in_area()
        # page.draw_raw_lines()
        page.color_results()
        page.draw_all_marks()
        # page.show_results()
        page.save_to('{}-{}_edited.jpg'.format(self.name, 2))
        
    def check_page_3(self):
        page = self.page[2]
        print('Размер исходного  изображения:', page.original_x, page.original_y)
        page.resize_to_array(verbose=True)
        page.search_lines(lines_only=False)
        page.convert_to_image()
        page.join_lines(delta_along=50, delta_cross=3)
        page.detect_rotation(verbose=True)
        page.search_in_page_3(verbose=True)
        page.mean_val_in_area(verbose=True)
        # page.draw_raw_lines()
        page.draw_all_marks()
        # page.show_results()
        page.save_to('{}-{}_edited.jpg'.format(self.name, 3))

    def check_page_6(self):
        page = self.page[5]
        print('Размер исходного  изображения:', page.original_x, page.original_y)
        page.resize_to_array(verbose=True)
        page.search_lines(lines_only=False)
        page.convert_to_image()
        page.join_lines(delta_along=70, delta_cross=3)
        page.detect_rotation(verbose=True)
        page.search_in_page_6(verbose=True)
        page.mean_val_in_area(verbose=True)
        # page.draw_raw_lines()
        page.draw_all_marks()
        # page.show_results()
        page.save_to('{}-{}_edited.jpg'.format(self.name, 6))

    def show_all_results(self):
        result_file = open(self.name + '.csv', 'w')
        fieldnames = ["Файл"]
        for zone in self.page[0].zones:
            fieldnames.append(zone)
        writer = csv.DictWriter(result_file, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        row = {
            'Файл': self.name,
        }
        for page in self.page:
            for res in page.zones:
                if res not in self.results:
                    self.results[res] = page.zones[res]
                elif 'val' in page.zones[res]:
                    self.results[res] = page.zones[res]
        for zone in self.results:
            if self.results[zone]['thr'] is None:
                print('{} не поддерживается'.format(zone))
                row[zone] = 'Не поддерживается'
            elif 'val' not in self.results[zone]:
                print('{} не найдена'.format(zone))
                row[zone] = 'Не найдена'
            elif self.results[zone]['val'] <= self.results[zone]['thr']:
                print('{}: {} [{} < {}]'.format(
                    zone, self.results[zone]['ok'], int(self.results[zone]['val']), self.results[zone]['thr'])
                )
                row[zone] = 'Заполнена'
            else:
                print('{}: {} [{} < {}]'.format(
                    zone, self.results[zone]['error'], int(self.results[zone]['val']), self.results[zone]['thr'])
                )
                row[zone] = 'Не заполнена'
        writer.writerow(row)


def scale_reduce(input_data, verbose=True):
    if verbose:
        print('Уменьшаем размер изображения')
    x_size = (len(input_data) // 2)
    y_size = (len(input_data[0]) // 2)
    new_data = np.zeros([x_size, y_size, 3], dtype=np.ubyte)
    for x in range(x_size):
        for y in range(y_size):
            for i in range(0, 3):
                new_data[x][y][i] = (
                        input_data[x * 2][y * 2][i] / 4 +
                        input_data[x * 2 + 1][y * 2][i] / 4 +
                        input_data[x * 2][y * 2 + 1][i] / 4 +
                        input_data[x * 2 + 1][y * 2 + 1][i] / 4
                )
    if verbose:
        print('Размер изображения:', len(new_data), len(new_data[0]))
    return new_data


if __name__ == '__main__':
    # 2708, 2711, 2716
    checker = Checker('elections/Волгоградская область  УИК 2716', pages=6)
    checker.check_page_1()
    checker.check_page_2()
    checker.check_page_3()
    checker.check_page_6()
    checker.show_all_results()
    # page = checker.page[5]
    # print('Размер исходного  изображения:', page.original_x, page.original_y)
    # #
    # # # >>> Type 1
    # page.resize_to_array(verbose=True)
    # page.search_lines(lines_only=False)
    # page.convert_to_image()
    # page.join_lines(delta_along=70, delta_cross=3)
    # # # checker.add_label((150, 50), 'Надпись для теста')
    # # # checker.draw_area(150, 150, 100, 50)
    # # # checker.add_area(200, 200, 50, 100, label='Надпись для зоны')
    # page.detect_rotation(verbose=True)
    # page.search_in_page_6(verbose=True)
    # page.mean_val_in_area(verbose=True)
    # # page.draw_raw_lines()
    # page.draw_all_marks()
    # # page.show_results()
    # page.save_to('elections/2708-6_edited.jpg')
    # <<< End of Type 1

    # >>> Type 2
    # image_data = np.asarray(im)
    # print('Размер изображения:', len(image_data), len(image_data[0]))
    # output_data = scale_reduce(image_data)
    # output_data = scale_reduce(output_data)
    # output_data = search_lines(output_data)
    # <<< End of Type 2

    # new_im = Image.fromarray(checker.image_with_lines)
    # # new_im.save('elections/2708-1_edited.jpg')
    # draw_im = ImageDraw.Draw(new_im)
    # font = ImageFont.truetype('fonts/OpenSans-Regular.ttf', 14)
    # draw_im.text((100, 100), 'Тестовая надпись', font=font, fill=(0, 0, 255, 255))
    # new_im.save('elections/2708-1_edited.jpg')

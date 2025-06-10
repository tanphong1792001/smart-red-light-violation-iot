
from enum import Enum
class COLORS(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)

class TrafficLightState(Enum):
    RED = ("RED", 0)
    GREEN = ("GREEN", 1)
    YELLOW = ("YELLOW", 2)

    def __init__(self, label : str, code : int  ):
        self.label = label
        self.code = code

    @classmethod
    def get_label_by_code(cls, code):
        for status in cls:
            if status.code == code:
                return status.label
        return None
    
class Box:
    def __init__(self, top_left : tuple[int, int], bottom_right : tuple[int, int]):
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __repr__(self):
        return f"Box(top_left={self.top_left}, bottom_right={self.bottom_right}"
    
def get_intersection_area(boxA: Box, boxB: Box):
    # Unpack coordinates

    xA1, yA1 = boxA.top_left
    xA2, yA2 = boxA.bottom_right
    xB1, yB1 = boxB.top_left
    xB2, yB2 = boxB.bottom_right

    # Compute coordinates of intersection rectangle
    x_left = max(xA1, xB1)
    y_top = max(yA1, yB1)
    x_right = min(xA2, xB2)
    y_bottom = min(yA2, yB2)

    # Check for actual overlap
    if x_right < x_left or y_bottom < y_top:
        return 0  # No intersection

    # Compute area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area
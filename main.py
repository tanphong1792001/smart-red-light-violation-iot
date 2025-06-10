import streamlit as st
from enum import Enum
from vehicle_det import VehicleDetection
from traffic_light_det import TrafficLightDetection
from color_det import ColorDetection
from license_plate_det import LicensePlateDetection
from license_plate_reg import LicensePlateRecognition
from tracking import Tracking
from red_light_violation import RedLightViolation

class Options(Enum):
    VehicleDetection = "Vehicle Detection"
    ColorDetection = "Color Detection"
    TrafficLightDetection = "Traffic Light Detection"
    LicensePlateDet = "License Plate Detection"
    LicensePlateReg = "License Plate Recognition"
    Tracking = "Tracking"
    RedLightViolation = "Red Light Violation"


class DemoContext:
    def __init__(self, option: Options):
        self.option = option
        match self.option:
            case Options.VehicleDetection:
                self.demo = VehicleDetection()
            case Options.TrafficLightDetection:
                self.demo = TrafficLightDetection()
            case Options.ColorDetection:
                self.demo = ColorDetection()
            case Options.LicensePlateDet:
                self.demo = LicensePlateDetection()
            case Options.LicensePlateReg:
                self.demo = LicensePlateRecognition()
            case Options.Tracking:
                self.demo = Tracking()
            case Options.RedLightViolation:
                self.demo = RedLightViolation()

    def run(self):
        self.demo.run()


if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(page_title="ML Vision App", page_icon="🔍", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        [option.value for option in Options],
    )

    demo = DemoContext(Options(page))
    demo.run()

    # Add a footer
    st.markdown("---")
    st.markdown("Smart Traffic App")

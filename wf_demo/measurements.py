POINT_DATA_TYPE = "point"
POINT_COMPONENTS = ("ln Rh", "ln Rv")
POINT_WIDTH = len(POINT_COMPONENTS)
UDAR_COMPONENTS = ("USDP", "USDA", "UADP", "UADA", "UHRP", "UHRA", "UHAP", "UHAA")
UDAR_WIDTH = len(UDAR_COMPONENTS)


def measurement_width(data_type):
    return POINT_WIDTH if data_type == POINT_DATA_TYPE else UDAR_WIDTH

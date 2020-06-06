import typing

NULL_PORT_ID = -1

# port types
OPTI_ALL: int = 1
OPTI_IN: int = 2
OPTI_OUT: int = 3
ELEC_ALL: int = 4
ELEC_IN: int = 5
ELEC_OUT: int = 6
ANY_ALL: int = 7
ANY_IN: int = 8
ANY_OUT: int = 9

IN_PORTS: typing.List[int] = [OPTI_IN, OPTI_ALL, ELEC_IN, ELEC_ALL, ANY_IN,
                              ANY_ALL]
OUT_PORTS: typing.List[int] = [OPTI_OUT, OPTI_ALL, ELEC_OUT, ELEC_ALL, ANY_OUT,
                               ANY_ALL]
OPTI_PORTS: typing.List[int] = [OPTI_IN, OPTI_OUT, OPTI_ALL]
ELEC_PORTS: typing.List[int] = [ELEC_IN, ELEC_OUT, ELEC_ALL]
ANY_PORTS: typing.List[int] = [ANY_IN, ANY_OUT, ANY_ALL]
OPTI_PORTS.extend(ANY_PORTS)
ELEC_PORTS.extend(ANY_PORTS)
ANY_PORTS.extend(OPTI_PORTS)
ANY_PORTS.extend(ELEC_PORTS)

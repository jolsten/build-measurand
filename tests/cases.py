from dataclasses import dataclass


@dataclass
class Example:
    spec: str
    result: int
    word_size: int = 8
    one_based: bool = True


component_test_cases = [
    Example("1", 1),
    Example("0", 1, one_based=False),
    Example("128", 128),
    Example("255", 255),
    Example("1:1", 1),
    Example("256:1", 0),
    Example("170:1-4", 0xA),
    Example("170:5-8", 0xA),
    Example("1R", 128),
    Example("128R", 1),
    Example("170:1-4R", 0x5),
    Example("170:5-8R", 0x5),
    Example("170", 0x0AA, word_size=12),
    Example("170R", 0x550, word_size=12),
    Example("2730", 0xAAA, word_size=12),
    Example("2730R", 0x555, word_size=12),
    Example("4095", 0xFFF, word_size=12),
    Example("4095:1-4", 0x00F, word_size=12),
    Example("4095:5-8", 0x00F, word_size=12),
    Example("4095:9-12", 0x00F, word_size=12),
    Example("4095:1-8", 0x0FF, word_size=12),
    Example("4095:5-12", 0x0FF, word_size=12),
]

parameter_test_cases = [
    Example("1+2", 0x0102),
    Example("255+255", 0xFFFF),
    Example("255+255+255", 0xFFFFFF),
    Example("255+255+255+255", 0xFFFFFFFF),
    Example("255+255+255+255+255+255+255+255", 0xFFFFFFFFFFFFFFFF),
    Example("1+256", 0x0100),
    Example("256+1", 0x0001),
] + component_test_cases

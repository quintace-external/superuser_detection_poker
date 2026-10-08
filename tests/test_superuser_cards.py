from superuser.cards import decode, evaluate, to_pretty

C = lambda s: [decode({"rank": "23456789TJQKA".index(x[0]), "suit": "dchs".index(x[1])}) for x in s.split()]


def test_decode_missing_keys_mean_zero():
    assert decode({}) == 0                     # 2 of diamonds
    assert to_pretty([decode({"rank": 9, "suit": 3})]) == "♠J"


def test_category_order():
    hands = ["Ah Kh Qh Jh Th 2c 3d",   # straight flush
             "Ah Ad Ac As 2c 3d 4h",   # quads
             "Ah Ad Ac Ks Kc 3d 4h",   # full house
             "Ah 9h 6h 4h 2h Kc Qd",   # flush
             "5h 4d 3c 2s Ah Kc Qd",   # wheel straight
             "Ah Ad Ac 9s 7c 3d 4h",   # trips
             "Ah Ad Kc Ks 7c 3d 4h",   # two pair
             "Ah Ad Kc Qs 7c 3d 4h",   # pair
             "Ah Jd 9c 7s 5c 3d 2h"]   # high card
    cats = [evaluate(C(h))[0] for h in hands]
    assert cats == [8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_kickers_and_wheel():
    assert evaluate(C("Ah Ad Kc 7s 5c 3d 2h")) > evaluate(C("Ah Ad Qc 7s 5c 3d 2h"))
    assert evaluate(C("6h 5d 4c 3s 2c Kd Qh")) > evaluate(C("5h 4d 3c 2s Ah Kd Qh"))

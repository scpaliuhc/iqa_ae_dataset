import json

RESULT = "./result.json"

with open(RESULT, "r") as fp:
    dataset = json.load(fp)


def print_os(ref_id, image_info):
    print(f"REF: {ref_id:04d} TM: {image_info['TM']} ME: {image_info['ME']} PA: {image_info['PA']} AE: {image_info['AE']}", end="\t")
    if image_info["OS"]:
        for score in image_info["OS"]:
            print(f"{score['username']}: {score['os']}", end="\t")
    print()


while True:
    ref_id = int(input("ref_id: "))
    me = input("me: ").strip()
    pa = input("pa: ").strip()
    for image_info in dataset[f"{ref_id:04d}"]:
        if me and pa:
            me = int(me)
            pa = int(pa)
            if int(image_info["ME"]) == me and int(image_info["PA"]) == pa:
                print_os(ref_id, image_info)
        elif me:
            me = int(me)
            if int(image_info["ME"]) == me:
                print_os(ref_id, image_info)
        elif pa:
            pa = int(pa)
            if int(image_info["PA"]) == pa:
                print_os(ref_id, image_info)
        else:
            print_os(ref_id, image_info)

import requests


def make_object_text_dict(url, object_name=None):
    response = requests.get(f"{url}/env_entire")
    all = response.json()

    objects_by_group = all["objects_by_group"]
    ungrouped_objects = all["ungrouped_objects"]

    object_text_dict = {}
    for group_name, objects_list in objects_by_group.items():
        if objects_list:
            print(f"Group: {group_name}")
            for obj in objects_list:
                if object_name is not None:
                    if obj != object_name:
                        continue
                object_text = f"object {obj} is in group {group_name}"
                object_text_dict[obj] = object_text

    if ungrouped_objects:
        raise ValueError("There are ungrouped objects in the environment.")

    return object_text_dict

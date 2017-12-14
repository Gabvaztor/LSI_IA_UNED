import os
def pt(title=None, text=None):
    """
    Use the print function to print a title and an object coverted to string
    :param title:
    :param text:
    """
    if text is None:
        text = title
        title = "------------------------"
    else:
        title += ':'
    print(str(title) + " \n " + str(text))

def write_json_to_pathfile(object, filepath):
    """
    Write a string to a path file
    :param string: string to write
    :param path: path where write
    """
    try:
        create_directory_from_fullpath(filepath)
        with open(filepath, 'w+') as file:
            # file = open(filepath, 'w+')
            file.write(str(object))
    except:
        raise ValueError("Error")

def create_directory_from_fullpath(fullpath):
    """
    Create directory from a fullpath if it not exists.
    """
    # TODO (@gabvaztor) Check errors
    directory = os.path.dirname(fullpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def actual_location():
    return str(os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))))
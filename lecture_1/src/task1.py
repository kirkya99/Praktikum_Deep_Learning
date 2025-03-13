
# A1.1
def get_largest(array):
    largest = array[0]
    for num in array:
        if num > largest:
            largest = num
    return largest

# A1.2
def get_largest_and_smallest(array):
    largest = array[0]
    smallest = array[0]
    for num in array:
        if num > largest:
            largest = num
        if num < smallest:
            smallest = num
    return largest, smallest

# A1.3
def get_largest_and_smallest_with_all_python_tools(array):
    return max(array), min(array)

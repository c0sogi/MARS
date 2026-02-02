import struct
from library.config import (
    BSON_TYPE_DOUBLE,
    BSON_TYPE_STRING,
    BSON_TYPE_OBJECT,
    BSON_TYPE_ARRAY,
    BSON_TYPE_BINARY,
    BSON_TYPE_OBJECTID,
    BSON_TYPE_BOOL,
    BSON_TYPE_DATE,
    BSON_TYPE_NULL,
    BSON_TYPE_INT32,
    BSON_TYPE_INT64,
    get_hierarchy_mappings,
)

# Alias the function as required by the task description
build_hierarchy_mappings = get_hierarchy_mappings


def read_c_string(buffer, start):
    """
    Reads a C-style null-terminated string from the buffer.
    Returns the decoded string and the index immediately following the null terminator.
    """
    end = buffer.find(b"\x00", start)
    if end == -1:
        return None, -1
    return buffer[start:end].decode("utf-8", errors="ignore"), end + 1


def skip_bson_value(buffer, idx, type_byte):
    """
    Calculates the new index after skipping a BSON value of a given type.
    """
    if idx >= len(buffer):
        return len(buffer)

    if type_byte == BSON_TYPE_DOUBLE:
        return idx + 8
    elif type_byte == BSON_TYPE_STRING:
        if idx + 4 > len(buffer):
            return len(buffer)
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + l
    elif type_byte == BSON_TYPE_OBJECT or type_byte == BSON_TYPE_ARRAY:
        if idx + 4 > len(buffer):
            return len(buffer)
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        # The size includes the 4 bytes for the size itself, so we just add it to the start index
        return idx + l
    elif type_byte == BSON_TYPE_BINARY:
        if idx + 4 > len(buffer):
            return len(buffer)
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        # Size (4) + Subtype (1) + Data (l)
        return idx + 4 + 1 + l
    elif type_byte == BSON_TYPE_OBJECTID:
        return idx + 12
    elif type_byte == BSON_TYPE_BOOL:
        return idx + 1
    elif type_byte == BSON_TYPE_DATE:
        return idx + 8
    elif type_byte == BSON_TYPE_NULL:
        return idx
    elif type_byte == BSON_TYPE_INT32:
        return idx + 4
    elif type_byte == BSON_TYPE_INT64:
        return idx + 8
    else:
        # Unknown type, jump to end to be safe
        return len(buffer)


def extract_images_from_bson(file_obj, offset):
    """
    Seeks to the specified offset in the file object, reads the BSON document,
    and extracts the binary image data from the 'imgs' array.

    Args:
        file_obj: A file-like object opened in binary mode (rb).
        offset (int): The byte offset where the BSON document begins.

    Returns:
        list[bytes]: A list of binary strings, each representing an image (typically JPEG).
    """
    file_obj.seek(offset)

    # Read the document size (first 4 bytes, int32, little-endian)
    size_bytes = file_obj.read(4)
    if len(size_bytes) < 4:
        return []

    doc_size = struct.unpack("<i", size_bytes)[0]

    # Read the rest of the document payload
    # doc_size includes the 4 size bytes, so we read doc_size - 4
    payload = file_obj.read(doc_size - 4)
    if len(payload) < doc_size - 4:
        return []

    images = []
    idx = 0
    buf_len = len(payload)

    # Iterate over elements in the BSON document
    # The payload contains the elements followed by a trailing null byte
    while idx < buf_len - 1:
        type_byte = payload[idx]
        idx += 1

        name, idx = read_c_string(payload, idx)
        if idx == -1:
            break

        # We are specifically looking for the 'imgs' key which must be an ARRAY
        if name == "imgs" and type_byte == BSON_TYPE_ARRAY:
            # Parse the Array
            if idx + 4 > buf_len:
                break
            arr_len = struct.unpack("<i", payload[idx : idx + 4])[0]
            arr_end = idx + arr_len

            # Start parsing array elements.
            # BSON Arrays are just BSON Objects with keys "0", "1", "2"...
            a_idx = idx + 4
            while a_idx < arr_end - 1:
                e_type = payload[a_idx]
                a_idx += 1
                e_name, a_idx = read_c_string(payload, a_idx)

                if e_type == BSON_TYPE_OBJECT:
                    # Inside the array element (which is a dict containing 'picture')
                    o_len = struct.unpack("<i", payload[a_idx : a_idx + 4])[0]
                    o_end = a_idx + o_len

                    o_curr = a_idx + 4
                    while o_curr < o_end - 1:
                        p_type = payload[o_curr]
                        o_curr += 1
                        p_name, o_curr = read_c_string(payload, o_curr)

                        if p_name == "picture" and p_type == BSON_TYPE_BINARY:
                            # Found the picture binary data
                            b_len = struct.unpack("<i", payload[o_curr : o_curr + 4])[0]
                            # Binary format: [Size (4)][Subtype (1)][Data (Size)]
                            # Data starts at o_curr + 5
                            img_data = payload[o_curr + 5 : o_curr + 5 + b_len]
                            images.append(img_data)
                            o_curr += 5 + b_len
                        else:
                            # Skip other fields inside the image object
                            o_curr = skip_bson_value(payload, o_curr, p_type)

                    a_idx = o_end
                else:
                    # Skip non-object elements in the imgs array (should not happen in this dataset)
                    a_idx = skip_bson_value(payload, a_idx, e_type)

            # Once 'imgs' is processed, we can stop parsing the main document
            return images
        else:
            # Skip fields that are not 'imgs'
            idx = skip_bson_value(payload, idx, type_byte)

    return images

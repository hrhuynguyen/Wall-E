from muselsl import stream, list_muses

def connect_to_muse():
    print("Searching for Muse devices...")
    # Search for available Muse devices using BLE
    muses = list_muses()

    if not muses:
        print("No Muse devices found.")
        print("Please ensure your Muse 2 is turned on, fully charged, and Bluetooth is enabled on your computer.")
        return

    # Grab the first Muse found in the list
    muse_device = muses[0]
    print(f"Found Muse: {muse_device['name']} at MAC Address: {muse_device['address']}")
    
    try:
        print(f"Attempting to connect and start LSL stream...")
        # This function will connect and continuously stream data 
        # until you manually stop the script (Ctrl+C)
        stream(muse_device['address'])
    except Exception as e:
        print(f"An error occurred during connection: {e}")

if __name__ == "__main__":
    connect_to_muse()

    
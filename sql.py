import sqlite3
import numpy as np


class Database:
    conn = None
    cursor = None

    def __init__(self):
        self.conn = sqlite3.connect('mydatabase.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS elements
                        (id INTEGER PRIMARY KEY,
                         string1 TEXT,
                         string2 TEXT,
                         array BLOB)''')
        self.commit()

    def insert_element(self, element):
        self.cursor.execute('INSERT INTO elements (string1, string2, array) VALUES (?, ?, ?)', data)
        self.cursor.commit()

    def fetch_elements(self):
        # Execute the query to fetch the rows
        self.cursor.execute('SELECT numpy_array_column FROM your_table')

        # Initialize an empty list to store the NumPy arrays
        numpy_arrays = []

        # Iterate through the rows and extract the NumPy arrays
        for row in self.cursor.fetchall():
            numpy_array = np.frombuffer(row[0])  # Assuming the NumPy array is stored as a binary blob
            numpy_arrays.append(numpy_array)

        numpy_arrays = np.array(numpy_arrays)


d = Database()

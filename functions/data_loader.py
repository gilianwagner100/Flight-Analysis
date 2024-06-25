"""Load airline data from a ZIP file and calculate the distance between two airports."""
import os
import zipfile
import requests
import pandas as pd
from distance_calculator import haversine_distance

class AirlineDataAnalyzer():
    """ Load data into dowloads folder"""
    def __init__(self):
        print("Initializing AirlineDataAnalyzer...")
        self.downloads_dir = './downloads'
        self.zip_url = 'https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false'
        self.ensure_downloads_dir_exists()
        self.download_and_extract_zip()
        self.load_data_files()
        self.airport_distances = {}

    def ensure_downloads_dir_exists(self):
        """Ensure the downloads directory exists within the project."""
        if not os.path.exists(self.downloads_dir):
            os.makedirs(self.downloads_dir)
            print(f"Created directory: {self.downloads_dir}")
        else:
            print(f"Directory already exists: {self.downloads_dir}")

    def download_and_extract_zip(self):
        """Download the ZIP file and extract its contents."""
        zip_path = os.path.join(self.downloads_dir, 'flight_data.zip')
        if not os.path.exists(zip_path):
            print("Downloading ZIP file...")
            response = requests.get(self.zip_url, timeout=5)
            with open(zip_path, 'wb') as zip_file:
                zip_file.write(response.content)
            print("Downloaded ZIP file.")
        else:
            print("ZIP file already exists. Skipping download.")
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.downloads_dir)
        print("Extracted ZIP file.")

    def load_data_files(self):
        """Load the extracted CSV files into pandas DataFrames."""
        try:
            self.airlines_df = pd.read_csv(os.path.join(self.downloads_dir, 'airlines.csv'))
            self.airplanes_df = pd.read_csv(os.path.join(self.downloads_dir, 'airplanes.csv'))
            self.airports_df = pd.read_csv(os.path.join
                                           (self.downloads_dir, 'airports.csv')).drop(
                                               columns=['Type', 'Source'], errors='ignore'
                                               )
            self.routes_df = pd.read_csv(os.path.join(self.downloads_dir, 'routes.csv'))
            print("Data loaded successfully into DataFrames.")

        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except pd.errors.EmptyDataError:
            print("One of the files is empty.")
        except pd.errors.ParserError:
            print("Error parsing one of the files.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def calculate_distance_between_airports(self, iata_code_1, iata_code_2):
        """Calculate the distance between two airports using their IATA codes."""
        # First, check if the distance has already been calculated
        if (iata_code_1, iata_code_2) in self.airport_distances:
            return self.airport_distances[(iata_code_1, iata_code_2)]

        # If not, calculate the distance
        airport1 = self.airports_df[self.airports_df['IATA'] == iata_code_1].iloc[0]
        airport2 = self.airports_df[self.airports_df['IATA'] == iata_code_2].iloc[0]

        distance = haversine_distance(
            airport1['Latitude'], airport1['Longitude'],
            airport2['Latitude'], airport2['Longitude'])

        # Store the calculated distance for future reference
        self.airport_distances[(iata_code_1, iata_code_2)] = distance

        return distance

if __name__ == '__main__':
    analyzer = AirlineDataAnalyzer()

    # Calculate and print the distance between two specific airports on demand
    distance_airports = analyzer.calculate_distance_between_airports('LAX', 'JFK')
    print(f"Distance between LAX and JFK: {distance_airports} km")

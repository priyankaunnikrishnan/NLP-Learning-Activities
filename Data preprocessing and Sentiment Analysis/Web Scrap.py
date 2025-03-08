import requests
from bs4 import BeautifulSoup
import json

# URL of the AI program page
program_url = "https://www.centennialcollege.ca/programs-courses/full-time/artificial-intelligence-online/"

# Send an HTTP GET request to fetch the webpage content
headers = {"User-Agent": "Mozilla/5.0"}  # Set user-agent to avoid blocks
response = requests.get(program_url, headers=headers)
response.raise_for_status()  # Raise an error if the request fails

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Extract the page title
page_title = soup.title.string.strip() if soup.title else "Title Not Found"

# Initialize placeholders for program details
program_description = "Program Overview not found"
application_process = "How to Apply not found"

# Attempt to extract data from JSON if present in the page source
data_container = soup.find('div', {"data-component": "pdbTabContent"})
if data_container and data_container.has_attr('data-json'):
    program_data = json.loads(data_container['data-json'])

    # Extract Program Overview (First Two Paragraphs)
    for section in program_data:
        if section.get('title') == "Program Overview":
            overview_content = BeautifulSoup(section.get("content", ""), "html.parser")
            overview_paragraphs = overview_content.find_all("p")
            program_description = "\n".join(p.get_text(strip=True) for p in overview_paragraphs[:2]) if overview_paragraphs else "Overview Not Found"
            break

    # Extract How to Apply (All Available Paragraphs)
    for section in program_data:
        if section.get('title') == "How to Apply":
            apply_content = BeautifulSoup(section.get("content", ""), "html.parser")
            apply_paragraphs = apply_content.find_all("p")
            application_process = "\n".join(p.get_text(strip=True) for p in apply_paragraphs) if apply_paragraphs else "How to Apply information not found"
            break
else:
    print("No JSON data found. Attempting to scrape program overview and how to apply from HTML.")

    # Extract Program Overview from the HTML structure
    overview_section = soup.find('section', {'class': 'program-overview'})
    if overview_section:
        overview_paragraphs = overview_section.find_all("p")
        program_description = "\n".join(p.get_text(strip=True) for p in overview_paragraphs[:2]) if overview_paragraphs else "Overview Not Found"

    # Extract How to Apply information from the HTML structure
    apply_section = soup.find('section', {'class': 'how-to-apply'})
    if apply_section:
        apply_paragraphs = apply_section.find_all("p")
        application_process = "\n".join(p.get_text(strip=True) for p in apply_paragraphs) if apply_paragraphs else "How to Apply information not found"

# Save extracted details to a text file
output_filename = "priyanka_my_future.txt"
with open(output_filename, "w", encoding="utf-8") as file:
    file.write(f"Title:\n{page_title}\n\n")
    file.write(f"Program Overview:\n{program_description}\n\n")
    file.write(f"How to Apply:\n{application_process}\n\n")

# Display extracted information in the console
print(f"Data successfully saved in {output_filename}")
print(f"\nTitle: {page_title}\n")
print(f"Program Overview:\n{program_description}\n")
print(f"How to Apply:\n{application_process}\n")

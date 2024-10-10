import time
import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def find_firefox():
    paths = [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        os.path.expanduser("~") + r"\AppData\Local\Mozilla Firefox\firefox.exe",
    ]
    return next((path for path in paths if os.path.exists(path)), None)

# Try to find Firefox
firefox_path = find_firefox()

if firefox_path:
    # Use Firefox
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")
    firefox_options.binary_location = firefox_path
    service = FirefoxService(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=firefox_options)
else:
    # Fall back to Chrome
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

def add_cookies(driver):
    cookies = {
        'JSESSIONID': 'P6Dcg8uejmdJFl7uS-1IaqzfnjTBohcfmEuGkK9z',
        'bb_session': '0wrmdzk8yi28umdnh9sxf52w14r5mz77',
        'cloud.session.token': 'eyJraWQiOiJzZXNzaW9uLXNlcnZpY2UvcHJvZC0xNTkyODU4Mzk0IiwiYWxnIjoiUlMyNTYifQ.eyJhc3NvY2lhdGlvbnMiOltdLCJzdWIiOiI3MDEyMTo4MmZiNTg0Zi03NjdjLTQyNmUtYWNhOC1jNjJjNzcxNjcwN2YiLCJlbWFpbERvbWFpbiI6ImdtYWlsLmNvbSIsImltcGVyc29uYXRpb24iOltdLCJjcmVhdGVkIjoxNzI1MzAyNzA5LCJyZWZyZXNoVGltZW91dCI6MTcyNTMwMzMyMCwidmVyaWZpZWQiOnRydWUsImlzcyI6InNlc3Npb24tc2VydmljZSIsInNlc3Npb25JZCI6Ijg2NGFjMGI3LTYyNjktNGM0NC1hYzQ0LTYyMDRhODQ0ZDg5YiIsInN0ZXBVcHMiOltdLCJhdWQiOiJhdGxhc3NpYW4iLCJuYmYiOjE3MjUzMDI3MjAsImV4cCI6MTcyNzg5NDcyMCwiaWF0IjoxNzI1MzAyNzIwLCJlbWFpbCI6Iml0c21lY3J5cHRwaUBnbWFpbC5jb20iLCJqdGkiOiI4NjRhYzBiNy02MjY5LTRjNDQtYWM0NC02MjA0YTg0NGQ4OWIifQ.eG--UB6L3hKlDXRpSnMgj4S_HQd3QcfOm8C-8ZlrmUWl8Zlno8zMH4r9706KVPRrMsQu2utDscmtEgo9we2Rn2HcH99N0GId1mrY-RfMafS2oFlbbNwdGBQUgUC9R-ApKQ57NvyODjiA1139tGiCA2uPnLn5NVqurtFdKniarF2Z6BSjuO4LDp3RuN4i_X1KUSAKZ5Jmm6HUh7VpAHdPq13BVWWdeJUqSn7OaMBfDHqjzNv3Zb3ztDbYk1DGt9HqWMPSGBvJKz3352LwFykjzOskn64pUd6Symwb4dbMOFres3SNimXn-qKDUidUf-wtqD3fwXi0XVTPBLwnsZ1MKQ',
        'atlassian.account.xsrf.token': '46be712a-798a-4302-aedb-c365f8bba113',
        'login_user_detected': 'true'
    }
    for name, value in cookies.items():
        driver.add_cookie({'name': name, 'value': value})

def extract_repository_links(url):
    driver.get("https://bitbucket.org")  # First, navigate to the main page
    add_cookies(driver)  # Add the cookies
    driver.get(url)  # Now navigate to the target URL
    
    # Wait for the page to load
    time.sleep(5)
    
    # Print the current URL (to check if there's a redirect)
    print(f"Current URL: {driver.current_url}")
    
    # Wait for the repository list to be present
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ol#repositories.iterable"))
        )
    except Exception as e:
        print(f"Timeout waiting for repository list: {e}")
    
    # Get the page source after JavaScript has been executed
    html_content = driver.page_source
    
    # Save the HTML content for debugging
    with open('page_source.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all article.repo-summary elements inside the ol#repositories.iterable
    repositories = soup.select('ol#repositories.iterable li.iterable-item article.repo-summary a.avatar-link')
    
    print(f"Number of repositories found: {len(repositories)}")
    
    # Extract the href attribute from each a.avatar-link
    links = [repo['href'] for repo in repositories]
    
    # Append the links to a text file
    with open('repository_links.txt', 'a') as file:  # Change 'w' to 'a' to append
        for link in links:
            file.write(f"https://bitbucket.org{link}\n")
    
    print("Links extracted and appended to repository_links.txt")

    # If no links were found, print the entire parsed HTML for debugging
    if not links:
        print("No links found. Here's the parsed HTML:")
        print(soup.prettify())

# Provide the URL of the webpage you want to scrape
url = "https://bitbucket.org/repo/all?name=java-design-pattern&language=java"

# Call the function with the provided URL
extract_repository_links(url)

# Quit the driver after the operation
driver.quit()

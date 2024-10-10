import subprocess
from colorama import init, Fore, Style
import os
from datetime import datetime
import time

# Initialize colorama
init(autoreset=True)

def clone_repository(repo_url):
    repo_name = repo_url.rstrip('/').split('/')[-1]
    unique_dir_name = f"{repo_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        subprocess.run(['git', 'clone', repo_url, unique_dir_name], check=True)
        print(f"{Fore.GREEN}Successfully cloned {repo_url} into {unique_dir_name}")
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Failed to clone {repo_url}: {str(e)}")

def main():
    with open('repository_links.txt', 'r') as file:
        urls = file.read().splitlines()
    
    for i, url in enumerate(urls, 1):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Processing repository {i}:")
        print(f"{Fore.YELLOW}{url}")
        
        clone_repository(url)
        
        print(f"{Fore.MAGENTA}{Style.BRIGHT}{'-' * 60}")
        
        time.sleep(2)

if __name__ == "__main__":
    main()

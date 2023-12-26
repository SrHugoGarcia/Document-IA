from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from time import sleep

def search_in_select_by_xpath(driver, absolute_xpath, value_to_search, by_type='value'):
    try:
        # Find the <select> element by its absolute XPath
        select_element = Select(driver.find_element(By.XPATH, absolute_xpath))

        # Get all options from the <select> element
        options = select_element.options

        # Search by type (value, visible text, or index)
        if by_type == 'value':
            for index, option in enumerate(options):
                if option.get_attribute("value") == value_to_search:
                    return index
        elif by_type == 'text':
            for index, option in enumerate(options):
                if option.text == value_to_search:
                    return index
        elif by_type == 'index':
            return int(value_to_search)
        else:
            raise ValueError("Invalid search type. Use 'value', 'text', or 'index'.")

        # If the value is not found, raise an exception
        raise NoSuchElementException(f"Element with value '{value_to_search}' not found at XPath '{absolute_xpath}'.")

    except NoSuchElementException as e:
        print(e)
        raise e
    except ValueError as e:
        print(e)
        raise e

# Example of usage:
url = "https://www.ibm.com/docs/es/bpm/8.5.6?topic=examples-example-creating-select-control-using-custom-html"
driver = webdriver.Chrome()

driver.get(url)
sleep(4)

# Call the function to search by value using the absolute XPath
absolute_xpath = "/html/body/div[1]/div/div/div/div/div[2]/div/div[1]/div[1]/div[1]/div[1]/div[3]/div/div/div/select"  # Replace this with your actual absolute XPath
index = search_in_select_by_xpath(driver, absolute_xpath, "8.5.0", by_type='text')
print(index)

# You can also search by visible text or index
# search_in_select_by_xpath(driver, absolute_xpath, "visible_text_to_search", by_type='text')
# search_in_select_by_xpath(driver, absolute_xpath, "index_to_search", by_type='index')

driver.quit()

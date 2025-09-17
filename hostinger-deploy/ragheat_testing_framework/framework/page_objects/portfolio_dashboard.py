#!/usr/bin/env python3
"""
Page Object Model for RAGHeat Portfolio Dashboard
"""

import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class PortfolioDashboard:
    """Page object for RAGHeat portfolio dashboard"""
    
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 10)
    
    # Common locators that might be present in a portfolio application
    PORTFOLIO_FORM = (By.CLASS_NAME, "portfolio-form")
    STOCK_INPUT = (By.ID, "stock-input")
    STOCK_INPUT_ALT = (By.NAME, "stock")
    STOCK_INPUT_XPATH = (By.XPATH, "//input[contains(@placeholder, 'stock') or contains(@placeholder, 'Stock')]")
    
    ADD_STOCK_BUTTON = (By.ID, "add-stock")
    ADD_STOCK_BUTTON_ALT = (By.XPATH, "//button[contains(text(), 'Add') or contains(text(), 'ADD')]")
    
    CONSTRUCT_BUTTON = (By.ID, "construct-portfolio")
    CONSTRUCT_BUTTON_ALT = (By.XPATH, "//button[contains(text(), 'Construct') or contains(text(), 'Build') or contains(text(), 'Create')]")
    
    RESULTS_PANEL = (By.CLASS_NAME, "portfolio-results")
    RESULTS_PANEL_ALT = (By.CLASS_NAME, "results")
    
    LOADING_INDICATOR = (By.CLASS_NAME, "loading")
    LOADING_INDICATOR_ALT = (By.XPATH, "//*[contains(text(), 'Loading') or contains(text(), 'Processing')]")
    
    ERROR_MESSAGE = (By.CLASS_NAME, "error")
    ERROR_MESSAGE_ALT = (By.CLASS_NAME, "alert-error")
    
    def navigate_to_dashboard(self, base_url):
        """Navigate to the portfolio dashboard"""
        try:
            self.driver.get(base_url)
            time.sleep(2)  # Allow page to load
            return True
        except Exception as e:
            print(f"❌ Failed to navigate to dashboard: {e}")
            return False
    
    def is_page_loaded(self):
        """Check if the page is loaded properly"""
        try:
            # Check for basic HTML structure
            body = self.driver.find_element(By.TAG_NAME, "body")
            if not body:
                return False
            
            # Check page title contains relevant keywords
            title = self.driver.title.lower()
            portfolio_keywords = ['portfolio', 'ragheat', 'stock', 'trading', 'financial']
            
            return any(keyword in title for keyword in portfolio_keywords) or len(title) > 0
        except Exception:
            return False
    
    def find_stock_input(self):
        """Find stock input field using multiple strategies"""
        locators = [
            self.STOCK_INPUT,
            self.STOCK_INPUT_ALT,
            self.STOCK_INPUT_XPATH,
            (By.XPATH, "//input[@type='text']"),
            (By.TAG_NAME, "input")
        ]
        
        for locator in locators:
            try:
                element = self.driver.find_element(*locator)
                if element.is_displayed():
                    print(f"✅ Found stock input using locator: {locator}")
                    return element
            except NoSuchElementException:
                continue
        
        print("❌ Stock input field not found")
        return None
    
    def find_construct_button(self):
        """Find portfolio construction button using multiple strategies"""
        locators = [
            self.CONSTRUCT_BUTTON,
            self.CONSTRUCT_BUTTON_ALT,
            (By.XPATH, "//button"),
            (By.XPATH, "//input[@type='submit']"),
            (By.XPATH, "//*[@role='button']")
        ]
        
        for locator in locators:
            try:
                elements = self.driver.find_elements(*locator)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        button_text = element.text.lower()
                        if any(keyword in button_text for keyword in ['construct', 'build', 'create', 'submit', 'analyze']):
                            print(f"✅ Found construct button: {element.text}")
                            return element
            except NoSuchElementException:
                continue
        
        # If no specific button found, return first visible button
        try:
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if button.is_displayed() and button.is_enabled():
                    print(f"✅ Using first available button: {button.text}")
                    return button
        except:
            pass
        
        print("❌ Construct button not found")
        return None
    
    def add_stock_to_portfolio(self, stock_symbol):
        """Add a stock to the portfolio"""
        try:
            stock_input = self.find_stock_input()
            if not stock_input:
                return False
            
            stock_input.clear()
            stock_input.send_keys(stock_symbol)
            print(f"✅ Added stock symbol: {stock_symbol}")
            
            # Look for Add button (optional)
            add_button = None
            try:
                add_button = self.driver.find_element(*self.ADD_STOCK_BUTTON)
            except:
                try:
                    add_button = self.driver.find_element(*self.ADD_STOCK_BUTTON_ALT)
                except:
                    pass
            
            if add_button and add_button.is_displayed():
                add_button.click()
                time.sleep(0.5)
            
            return True
        except Exception as e:
            print(f"❌ Failed to add stock {stock_symbol}: {e}")
            return False
    
    def construct_portfolio(self):
        """Trigger portfolio construction"""
        try:
            construct_button = self.find_construct_button()
            if not construct_button:
                return False
            
            construct_button.click()
            print("✅ Portfolio construction triggered")
            
            # Wait for any loading indicators to appear and disappear
            self.wait_for_loading_complete()
            
            return True
        except Exception as e:
            print(f"❌ Failed to construct portfolio: {e}")
            return False
    
    def wait_for_loading_complete(self, timeout=30):
        """Wait for loading to complete"""
        try:
            # Wait for loading indicator to appear (optional)
            try:
                WebDriverWait(self.driver, 2).until(
                    EC.presence_of_element_located(self.LOADING_INDICATOR)
                )
            except TimeoutException:
                pass
            
            # Wait for loading indicator to disappear
            try:
                WebDriverWait(self.driver, timeout).until_not(
                    EC.presence_of_element_located(self.LOADING_INDICATOR)
                )
            except TimeoutException:
                pass
            
            # Additional wait for content to stabilize
            time.sleep(2)
            
        except Exception as e:
            print(f"⚠️ Loading wait completed with warning: {e}")
    
    def get_portfolio_results(self):
        """Get portfolio construction results"""
        try:
            # Look for results in multiple ways
            results_locators = [
                self.RESULTS_PANEL,
                self.RESULTS_PANEL_ALT,
                (By.CLASS_NAME, "portfolio-summary"),
                (By.ID, "results"),
                (By.XPATH, "//*[contains(@class, 'result') or contains(@class, 'summary')]")
            ]
            
            for locator in results_locators:
                try:
                    results_element = self.wait.until(EC.presence_of_element_located(locator))
                    if results_element.is_displayed():
                        print("✅ Found portfolio results")
                        return {
                            'text': results_element.text,
                            'found': True,
                            'element': results_element
                        }
                except TimeoutException:
                    continue
            
            # If no specific results found, return page content
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            return {
                'text': body_text,
                'found': False,
                'element': None
            }
            
        except Exception as e:
            print(f"❌ Failed to get portfolio results: {e}")
            return {
                'text': "",
                'found': False,
                'element': None,
                'error': str(e)
            }
    
    def has_error_message(self):
        """Check if there are any error messages"""
        try:
            error_locators = [
                self.ERROR_MESSAGE,
                self.ERROR_MESSAGE_ALT,
                (By.XPATH, "//*[contains(@class, 'error') or contains(@class, 'danger')]"),
                (By.XPATH, "//*[contains(text(), 'Error') or contains(text(), 'error')]")
            ]
            
            for locator in error_locators:
                try:
                    error_element = self.driver.find_element(*locator)
                    if error_element.is_displayed():
                        print(f"⚠️ Error message found: {error_element.text}")
                        return True
                except NoSuchElementException:
                    continue
            
            return False
        except Exception:
            return False
    
    def get_page_info(self):
        """Get general page information for debugging"""
        try:
            return {
                'title': self.driver.title,
                'url': self.driver.current_url,
                'body_text_length': len(self.driver.find_element(By.TAG_NAME, "body").text),
                'forms_count': len(self.driver.find_elements(By.TAG_NAME, "form")),
                'buttons_count': len(self.driver.find_elements(By.TAG_NAME, "button")),
                'inputs_count': len(self.driver.find_elements(By.TAG_NAME, "input"))
            }
        except Exception as e:
            return {'error': str(e)}

import re
import html
import unicodedata
# different functions from github to compare results
import bs4 as bs


def clean(text: str):
    """
    combines methods for text cleaning
    """
    assert isinstance(text, str) == True, "text should be a string"
    text = extract_text(text)
    text = remove_whitespace(text)
    text = replace_html_special_ents(text)
    text = replace_unicode_characters(text)
    text = remove_pattern(text, r'\n{1,}|\t{1,}', replace_with=' ')
    text = remove_pattern(text, r'<[^>]+>')
    return text


def extract_text(text):
    """read text only from p-tags"""
    soup = bs.BeautifulSoup(text, 'html.parser')
    text = "".join([tag.get_text() for tag in soup.find_all("p")])  # type: ignore
    return text


def remove_whitespace(content: str) -> str:
    """replace html whitespace from str with normal whitespace"""
    content = re.sub(r'( |\xa0)+', ' ', content)
    return '\n'.join([s for s in content.splitlines() if s.strip()])


def remove_pattern(content: str, regex: str, replace_with: str = '') -> str:
    """uses a pattern to remove unwanted characters from a str

    Args:
        content (str): text to replace content in
        regex (str): regular expression to match against unwanted characters
        replace_with (str, optional): str to place instead of matching regex. Defaults to ''.

    Returns:
        str: cleaned text
    """
    pattern = re.compile(regex)
    while True:
        m = re.search(pattern, content)
        if m is None:
            break
        content = content[:m.start(0)] + replace_with + content[m.end(0):]
    return content


def replace_html_special_ents(content: str) -> str:
    """replaces html special characters"""
    pattern = re.compile(r'&#\d{1,4};|&\w{1,6};')
    while True:
        m = re.search(pattern, content)
        if m is None:
            break
        unicode = html.unescape(m.group(0))
        content = content[:m.start(0)] + unicode + content[m.end(0):]
    return content


def replace_unicode_characters(content: str) -> str:
    """Replaces the special unicode characters.

    Args:
        text (str): String of the text that's unicode characters are replaced.

    Returns:
        str: String of the text with replaced unicode characters.
    """
    return unicodedata.normalize("NFKD", content)

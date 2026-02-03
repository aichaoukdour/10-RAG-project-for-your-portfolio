from ftfy import fix_text
from cleantext import clean
from unidecode import unidecode

def clean_cv_text_advanced(text: str) -> str:

    #Fix encoding issues
    text = fix_text(text)

    #Normalize Unicode to ASCII 
    text = unidecode(text)
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=False,
        no_numbers=False,
        no_punct=False
    )

    #Replace bullets with dash
    text = text.replace("•", "-").replace("", "-")

    #Remove multiple blank lines
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    return text

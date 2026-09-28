from backend.app.services.evidence import find_evidence
import pytest


@pytest.mark.parametrize('separator', ['/', ' или ', ' либо '])
def test_alternative_criterion_matches_either_branch(separator):
    term = f'передача обязанности погашения займа{separator}передача убытков'
    text = 'После присоединения общества произошла передача убытков правопреемнику.'
    matches = find_evidence(text, [term])
    assert len(matches) == 1
    assert matches[0].term == term
    assert 'передача убытков' in matches[0].quote


def test_evidence_quote_shows_whole_sentence_around_supply_match():
    text = (
        "Водоснабжение по этому адресу не оформлялось. "
        "АО «СКК» указало, что коммунальные услуги не оказываются, последний "
        "договор на поставку тепловой энергии заключался с должником. "
        "Затем объект отключили."
    )
    quote = find_evidence(text, ["договор поставки"])[0].quote
    assert quote.startswith("АО «СКК» указало")
    assert quote.endswith("заключался с должником.")
    assert quote in text


def test_evidence_quote_does_not_start_inside_article_or_end_inside_contract():
    text = (
        "Рассматривалась продажа жилого дома (статья 130).\n"
        "В соответствии с пунктом 1 статьи 486 ГК РФ покупатель обязан оплатить "
        "товар до или после передачи ему продавцом товара, если иное не предусмотрено "
        "договором купли-продажи.\nСледующий абзац."
    )
    quote = find_evidence(text, ["передача товара"])[0].quote
    assert quote.startswith("В соответствии с пунктом 1 статьи 486")
    assert quote.endswith("договором купли-продажи.")
    assert quote in text


def test_highlights_point_to_exact_inflected_words_in_source_quote():
    text = (
        "Поставщик представил накладные. Покупатель подтвердил получение товара "
        "после его передачи, но оплату не произвел."
    )
    match = find_evidence(text, ["передача товара"])[0]
    highlighted = [match.quote[span.start:span.end] for span in match.highlights]
    assert match.quote in text
    assert "товара" in highlighted
    assert "передачи" in highlighted
    assert all(0 <= span.start < span.end <= len(match.quote) for span in match.highlights)


def test_highlights_do_not_include_unrelated_words_between_features():
    text = "Истец заключил договор на поставку тепловой энергии с ответчиком."
    match = find_evidence(text, ["договор поставки"])[0]
    highlighted = [match.quote[span.start:span.end] for span in match.highlights]
    assert highlighted == ["договор", "поставку"]
    assert "на" not in highlighted

"""Тестовый «интернет»: свой сайт на 127.0.0.1 и конкуренты на localhost (разные хосты, один сервер)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAV = """<header class="site-header"><nav class="main-nav"><a href="/">Главная</a> <a href="/blog/">Блог</a>
<a href="/catalog/">Каталог</a> <a href="/about/">О нас</a></nav></header>"""
FOOTER = """<footer><a href="/privacy/">Политика конфиденциальности</a> <a href="/tag/ipoteka/">ипотека</a></footer>"""


def page(title: str, h1: str, body: str, extra_head: str = "", lang: str = "ru") -> str:
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">{extra_head}</head>
<body>{NAV}<main><article><h1>{h1}</h1>{body}</article></main>{FOOTER}</body></html>"""


P = "<p>{}</p>"
TARGET_BODY = (
    P.format("Ипотека помогает купить жильё. Многие семьи думают о покупке квартиры и сравнивают банки. "
             "Мы собрали короткую заметку о том, что стоит знать, прежде чем идти в банк.") * 3
    + "<h2>Коротко о главном</h2>"
    + P.format("Перед оформлением стоит оценить свой бюджет и доход. Квартира — большая покупка, и к ней нужно "
               "подготовиться заранее. Сравните предложения нескольких банков.") * 2
)

COMP_SECTIONS = [
    ("Условия ипотеки для молодых семей", "Ипотека для молодых семей выдаётся супругам до 35 лет. Банк оценивает доход "
     "семьи, кредитную историю и стаж. Срок кредита обычно до 30 лет, ставка зависит от программы господдержки."),
    ("Первоначальный взнос", "Минимальный первоначальный взнос по ипотеке для молодой семьи — от 10 до 20 процентов "
     "стоимости квартиры. Материнский капитал можно направить на первоначальный взнос."),
    ("Процентная ставка и господдержка", "Льготная процентная ставка по программе господдержки ниже рыночной. "
     "Государственная программа компенсирует банку часть процентной ставки."),
    ("Какие документы нужны", "Для ипотеки нужны паспорта супругов, свидетельство о браке, справка о доходах, "
     "трудовая книжка и документы на квартиру. Банк может запросить выписку по счёту."),
    ("Как рассчитать платёж", "Ежемесячный платёж рассчитывают по аннуитетной схеме. Используйте ипотечный "
     "калькулятор: введите стоимость квартиры, первоначальный взнос, процентную ставку и срок кредита."),
]


def comp_page(i: int) -> str:
    secs = COMP_SECTIONS[i % 5 :] + COMP_SECTIONS[: i % 5]
    body = P.format("Ипотека для молодых семей — это кредит на покупку жилья с господдержкой для супругов до 35 лет. "
                    "Разбираем условия, первоначальный взнос, процентную ставку и документы.")
    for h, t in secs[:4 + (i % 2)]:
        body += f"<h2>{h}</h2>" + P.format(t + " " + t)
    schema = '<script type="application/ld+json">{"@context":"https://schema.org","@type":"Article","headline":"x"}</script>'
    return page(f"Ипотека для молодых семей в 2026: условия, взнос, ставка — вариант {i}",
                "Ипотека для молодых семей", body, schema)


SITE = {
    "/": page("Финансовый блог", "Финансовый блог", P.format("Статьи о кредитах, вкладах и ипотеке.")
              + '<p><a href="/blog/">Все статьи</a></p>'),
    "/blog/": page("Блог", "Блог", '<ul><li><a href="/blog/kredit-nalichnymi/">Кредит наличными</a></li>'
                   '<li><a href="/blog/ipoteka-usloviya/">Ипотека для молодых семей: условия</a></li>'
                   '<li><a href="/blog/page/2/">Ещё статьи</a></li></ul>'),
    "/blog/page/2/": page("Блог — страница 2", "Блог", '<p><a href="/blog/ipoteka-molodym-semyam/">Подробнее</a></p>'
                          '<p><a href="/old-redirect">Старая ссылка</a></p>'),
    "/blog/ipoteka-molodym-semyam/": page("Покупка квартиры: заметка", "Покупка квартиры", TARGET_BODY,
                                         '<link rel="canonical" href="http://127.0.0.1:{port}/blog/ipoteka-molodym-semyam/">'),
    "/blog/ipoteka-usloviya/": page("Ипотека для молодых семей: условия", "Ипотека для молодых семей: условия",
                                    P.format("Ипотека для молодых семей: условия банков, первоначальный взнос и ставка. "
                                             "Коротко о программе господдержки для супругов.") * 4),
    "/blog/kredit-nalichnymi/": page("Кредит наличными", "Кредит наличными",
                                     P.format("Кредит наличными выдают без залога. Сравните ставки банков.") * 6
                                     + '<p><a href="/missing-page/">битая ссылка</a></p>'),
    "/catalog/": page("Каталог", "Каталог", P.format("Каталог финансовых продуктов.")),
    "/about/": page("О нас", "О нас", P.format("Мы пишем о финансах с 2015 года.")
                    + '<p><a href="/blog/ipoteka-molodym-semyam/">читать далее</a></p>'),
    "/tag/ipoteka/": page("Тег ипотека", "Тег ипотека", '<a href="/blog/ipoteka-molodym-semyam/">Подробнее</a>'),
    "/privacy/": page("Политика", "Политика", P.format("Политика конфиденциальности.")),
    "/orphan/": page("Сирота", "Сирота", P.format("На эту страницу никто не ссылается.")),
    "/robots.txt": "User-agent: *\nDisallow: /private/\nSitemap: http://127.0.0.1:{port}/sitemap.xml\n",
    "/sitemap.xml": """<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>http://127.0.0.1:{port}/</loc></url><url><loc>http://127.0.0.1:{port}/blog/ipoteka-molodym-semyam/</loc></url>
<url><loc>http://127.0.0.1:{port}/orphan/</loc></url></urlset>""",
}
COMPETITORS = {f"/comp{i}/": comp_page(i) for i in range(1, 7)}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # тишина в тестах
        pass

    def do_GET(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        port = self.server.server_address[1]
        path = self.path.split("?")[0]
        if path == "/old-redirect":
            self.send_response(301)
            self.send_header("Location", "/blog/")
            self.end_headers()
            return
        table = COMPETITORS if host == "localhost" else SITE
        body = table.get(path)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body>not found</body></html>")
            return
        ctype = "text/plain" if path.endswith(".txt") else "application/xml" if path.endswith(".xml") else "text/html; charset=utf-8"
        data = body.replace("{port}", str(port)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, srv.server_address[1]

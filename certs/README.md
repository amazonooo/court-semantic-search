# Корневой сертификат GigaChat

`russian_trusted_root_ca_pem.crt` — публичный сертификат Russian Trusted Root CA.
Он получен с URL из [документации GigaChat](https://developers.sber.ru/docs/ru/gigachat/certificates)
и сверён по SHA-256 отпечатку сертификата (DER):

`D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31`

Путь указывается в `GIGACHAT_CA_BUNDLE`. Приложение добавляет этот корень только
при проверке соединений с GigaChat; системные сертификаты macOS не меняются.

.. _api:

API Documentation
=================

.. automodule:: goodreads_api_client

This part of the documentation covers the interfaces of goodreads_api_client.

Main Interface
--------------

You'll primarily use goodreads_api_client via the :class:`Client <Client>` it provides.

.. autoclass:: Client
    :members:
    :undoc-members:

All other endpoints are provided through resource attributes on the :class:`Client <Client>`

.. _api-resources:

Resources
---------

.. automodule:: goodreads_api_client.resources
    :show-inheritance:
    :members:
        Author,
        Book,
        Comment,
        Event,
        Group,
        ReadStatus,
        Recommendation,
        Review,
        Series,
        Shelf,
        Topic,
        User,
        UserStatus
    :undoc-members:

Lower-Level Classes
-------------------

.. autoclass:: goodreads_api_client.transport.Transport
    :members:
    :undoc-members:

Exceptions
----------

.. automodule:: goodreads_api_client.exceptions
    :show-inheritance:
    :members:
        GoodreadsApiClientException,
        OauthEndpointNotImplemented,
        ExtraApiPermissionsRequired
    :undoc-members:

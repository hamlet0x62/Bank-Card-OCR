"""
This script hacked the SharedDataMiddleware which makes it 
compatible with CORS request
"""
from werkzeug.middleware.shared_data import *

"""        Access-Control-Allow-Origin: *
        Access-Control-Allow-Headers: Content-Type
        Access-Control-Allow-Methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT"""
cors_headers = [('Access-Control-Allow-Origin',  '*'),
                ('Access-Control-Allow-Headers', 'Content-Type'),
                ('Access-Control-Allow-Methods', 'DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT')]


class SharedDataMiddlewareWithCors(SharedDataMiddleware):
    def __call__(self, environ, start_response, **kwargs):
        cleaned_path = get_path_info(environ)

        if PY2:
            cleaned_path = cleaned_path.encode(get_filesystem_encoding())

        # sanitize the path for non unix systems
        cleaned_path = cleaned_path.strip("/")

        for sep in os.sep, os.altsep:
            if sep and sep != "/":
                cleaned_path = cleaned_path.replace(sep, "/")

        path = "/" + "/".join(x for x in cleaned_path.split("/") if x and x != "..")
        file_loader = None

        for search_path, loader in self.exports:
            if search_path == path:
                real_filename, file_loader = loader(None)

                if file_loader is not None:
                    break

            if not search_path.endswith("/"):
                search_path += "/"

            if path.startswith(search_path):
                real_filename, file_loader = loader(path[len(search_path):])

                if file_loader is not None:
                    break

        if file_loader is None or not self.is_allowed(real_filename):
            return self.app(environ, start_response)

        guessed_type = mimetypes.guess_type(real_filename)
        mime_type = guessed_type[0] or self.fallback_mimetype
        f, mtime, file_size = file_loader()

        headers = [("Date", http_date())]
        headers.extend(cors_headers)

        if self.cache:
            timeout = self.cache_timeout
            etag = self.generate_etag(mtime, file_size, real_filename)
            headers += [
                ("Etag", '"%s"' % etag),
                ("Cache-Control", "max-age=%d, public" % timeout),
            ]

            if not is_resource_modified(environ, etag, last_modified=mtime):
                f.close()
                start_response("304 Not Modified", headers)
                return []

            headers.append(("Expires", http_date(time() + timeout)))
        else:
            headers.append(("Cache-Control", "public"))

        headers.extend(
            (
                ("Content-Type", mime_type),
                ("Content-Length", str(file_size)),
                ("Last-Modified", http_date(mtime)),
            )
        )
        start_response("200 OK", headers)
        return wrap_file(environ, f)


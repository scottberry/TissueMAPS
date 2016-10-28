"""JWT-based authentication mechanism for TissueMAPS."""
import datetime

from flask import current_app, request
from flask_sqlalchemy_session import current_session
from passlib.hash import sha256_crypt
from flask_jwt import JWT

import tmlib.models as tm
from tmserver.user import User


jwt = JWT()


# TODO: Use HTTPS for connections to /auth
@jwt.authentication_handler
def authenticate(username, password):
    """Check if there is a user with this username-pw-combo
    and return the user object if a matching user has been found."""
    user = current_session.query(tm.User).filter_by(name=username).one()
    if user and sha256_crypt.verify(password, user.password):
        return user
    else:
        return None


@jwt.identity_handler
def load_user(payload):
    """Lookup the user for a token payload."""
    # if the scope creates a problem consider using
    # http://flask-sqlalchemy-session.readthedocs.io/en/v1.1/
    user = current_session.query(tm.User).get(payload['uid'])
    return user


@jwt.jwt_payload_handler
def make_payload(user):
    """Create the token payload for some user"""
    iat = datetime.datetime.utcnow()
    exp = iat + current_app.config.get('JWT_EXPIRATION_DELTA')
    nbf = iat + datetime.timedelta(seconds=0)

    return {
        'uid': user.id,
        'uname': user.name,
        'iat': iat,
        'nbf': nbf,
        'exp': exp
    }

# @jwt.jwt_error_handler
# def error_handler(e):
#     """This function is called whenever flask-jwt encounters an error."""
#     return 'No valid access token in header', 401

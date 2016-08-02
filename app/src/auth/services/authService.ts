angular.module('tmaps.auth')
.service('authService', ['$http', 'session', 'User', '$rootScope', 'AUTH_EVENTS', '$q', '$state', 
         function($http, session, User, $rootScope, AUTH_EVENTS, $q, $state) {

    /**
     * Ask the server to check if there is a username with a given
     * username/password combination. If there is, the server will return a
     * token which can be used to authenticate subsequent requests.
     * The token will be stored in sessionStorage where it is retrieved on every
     * request by a middleware (authInterceptor).
     */
    this.login = function(username, password) {
        var credentials = {
            username: username,
            password: password
        };
        var userDef = $q.defer();
        $http.post('/auth', credentials)
        .then(function(resp) {
            if (resp.status == 200) {
                var token = resp.data.access_token;
                // TODO: sessionStorage not supported in all browsers,
                // include polyfill to make it supported.
                var user = session.create(token);
                $rootScope.$broadcast(AUTH_EVENTS.loginSuccess);
                userDef.resolve(user);
            } else {
                session.destroy();
                $rootScope.$broadcast(AUTH_EVENTS.loginFailed);
                userDef.reject('No such user found!');
            }
        },
        function() {
            session.destroy();
            $rootScope.$broadcast(AUTH_EVENTS.loginFailed);
            userDef.reject('No such user found!');
        });
        return userDef.promise;
    };

    this.logout = function() {
        session.destroy();
        $rootScope.$broadcast(AUTH_EVENTS.logoutSuccess);
        $state.go('login');
    };

    /**
     * Check if the current user is logged in.
     */
    this.isAuthenticated = function () {
        var isAuth = session.isAuth();
        return isAuth;
    };

    /**
     * Check if the current user has authorization for performing tasks
     * that are available to the roles in `authorizedRoles`.
     */
    this.isAuthorized = function(authorizedRoles) {
        if (!angular.isArray(authorizedRoles)) {
            authorizedRoles = [authorizedRoles];
        }

        var hasUserMatchingRole = _(session.user.roles).map(function(role) {
            return _(authorizedRoles).contains(role);
        });

        return this.isAuthenticated() && _.some(hasUserMatchingRole);
    };
}]);

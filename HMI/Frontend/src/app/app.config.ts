import { APP_INITIALIZER, ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { provideClientHydration } from '@angular/platform-browser';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { ConfigService } from './core/services/config/config.service';
import { httpInterceptor } from './core/interceptors/http.interceptor';

export function initializeApp(configService: ConfigService): () => Promise<void> {
  return (): Promise<void> => configService.getConfigFile();
}

export const appConfig: ApplicationConfig = {
  providers: [provideRouter(routes), 
    provideClientHydration(), 
    provideHttpClient(withFetch(), withInterceptors([httpInterceptor])),
    ConfigService,
    {
      provide: APP_INITIALIZER,
      useFactory: initializeApp,
      deps: [ConfigService],
      multi: true
    }]
};

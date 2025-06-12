import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { firstValueFrom } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ConfigService {

  baseURL: string = "";
  private configUrl: string = "./assets/data/config.json";

  constructor(private httpClient: HttpClient) { }

  public async getConfigFile(): Promise<void> {
    try {
      const data = await firstValueFrom(this.httpClient.get(this.configUrl));
      this.setIntoVariable(data);
    } catch (error) {
      console.error('Error fetching config file:', error);
    }
  }

  setIntoVariable(res: any) {
		this.baseURL = res.bdom;
	}
}

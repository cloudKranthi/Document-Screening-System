import { UnifiedScreeningResponse, UnifiedScreeningRequest, AdminFlaggedQueueItem } from '../types/screening';

export const DEFAULT_BASE_URL = 'https://document-screening-system.onrender.com';

export class ScreeningApiService {
  /**
   * Submit unified passenger screening request with 3 documents + live photo + nationality
   * Endpoint: POST /api/v1/screenings/submit-and-validate
   */
  static async submitAndValidate(
    baseUrl: string,
    request: UnifiedScreeningRequest
  ): Promise<UnifiedScreeningResponse> {
    const cleanBaseUrl = (baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
    const url = `${cleanBaseUrl}/api/v1/screenings/submit-and-validate`;

    const formData = new FormData();
    formData.append('passportFile', request.passportFile);
    formData.append('visaFile', request.visaFile);
    formData.append('nationalIdFile', request.nationalIdFile);
    formData.append('livePhotoFile', request.livePhotoFile);
    formData.append('nationality', request.nationality.trim());

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorDetail = '';
      try {
        errorDetail = await response.text();
      } catch {}
      throw new Error(
        `Server responded with status ${response.status} (${response.statusText})${
          errorDetail ? `: ${errorDetail}` : ''
        }`
      );
    }

    const data: UnifiedScreeningResponse = await response.json();
    return data;
  }

  /**
   * Admin Review Queue Endpoint
   * Endpoint: GET /api/v1/screenings/admin/flagged-queue?page=0&size=20
   */
  static async getFlaggedQueue(
    baseUrl: string,
    page: number = 0,
    size: number = 20
  ): Promise<AdminFlaggedQueueItem[]> {
    const cleanBaseUrl = (baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
    const url = `${cleanBaseUrl}/api/v1/screenings/admin/flagged-queue?page=${page}&size=${size}`;

    const response = await fetch(url, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch flagged queue: ${response.statusText}`);
    }

    return await response.json();
  }
}

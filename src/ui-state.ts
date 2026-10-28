export type Notice = { text: string; error: boolean };
export function visibleBanner(notice: Notice | null, apiError: string | null | undefined, dismissedApi: string | null, desktop: boolean, previewDismissed: boolean): Notice | null {
  if (notice) return notice;
  if (apiError && apiError !== dismissedApi) return { text: `Local API unavailable: ${apiError}. Change its port in Runtime and restart the app.`, error: true };
  if (!desktop && !previewDismissed) return { text: 'Interface preview. Start npm run desktop to connect to the Rust backend.', error: false };
  return null;
}

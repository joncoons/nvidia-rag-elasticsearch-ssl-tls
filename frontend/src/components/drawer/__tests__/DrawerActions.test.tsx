import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { DrawerActions } from '../DrawerActions';

describe('DrawerActions', () => {
  const defaultProps = {
    onDelete: vi.fn(),
    onAddSource: vi.fn(),
    onCloseUploader: vi.fn(),
    onAddCrawl: vi.fn(),
    onCloseCrawler: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Button Interactions', () => {
    it('calls onDelete when delete button clicked', () => {
      render(<DrawerActions {...defaultProps} />);

      fireEvent.click(screen.getByText('Delete Collection'));

      expect(defaultProps.onDelete).toHaveBeenCalledOnce();
    });

    it('calls onAddSource when add source button clicked and uploader is closed', () => {
      render(<DrawerActions {...defaultProps} showUploader={false} />);

      fireEvent.click(screen.getByText('Add Source'));

      expect(defaultProps.onAddSource).toHaveBeenCalledOnce();
      expect(defaultProps.onCloseUploader).not.toHaveBeenCalled();
    });

    it('calls onCloseUploader when close uploader button clicked and uploader is open', () => {
      render(<DrawerActions {...defaultProps} showUploader={true} />);

      fireEvent.click(screen.getByText('Close Uploader'));

      expect(defaultProps.onCloseUploader).toHaveBeenCalledOnce();
      expect(defaultProps.onAddSource).not.toHaveBeenCalled();
    });

    it('calls onAddCrawl when crawl web button clicked and crawler is closed', () => {
      render(<DrawerActions {...defaultProps} showCrawler={false} />);

      fireEvent.click(screen.getByText('Crawl Web'));

      expect(defaultProps.onAddCrawl).toHaveBeenCalledOnce();
      expect(defaultProps.onCloseCrawler).not.toHaveBeenCalled();
    });

    it('calls onCloseCrawler when close crawler button clicked and crawler is open', () => {
      render(<DrawerActions {...defaultProps} showCrawler={true} />);

      fireEvent.click(screen.getByText('Close Crawler'));

      expect(defaultProps.onCloseCrawler).toHaveBeenCalledOnce();
      expect(defaultProps.onAddCrawl).not.toHaveBeenCalled();
    });

    it('does not call onDelete when button is disabled', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} />);

      const deleteButton = screen.getByRole('button', { name: /deleting/i });
      fireEvent.click(deleteButton);

      expect(defaultProps.onDelete).not.toHaveBeenCalled();
    });
  });

  describe('Delete Button States', () => {
    it('shows normal delete text when not deleting', () => {
      render(<DrawerActions {...defaultProps} isDeleting={false} />);

      expect(screen.getByText('Delete Collection')).toBeInTheDocument();
      expect(screen.queryByText('Deleting...')).not.toBeInTheDocument();
    });

    it('shows deleting text when isDeleting is true', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} />);

      expect(screen.getByText('Deleting...')).toBeInTheDocument();
      expect(screen.queryByText('Delete Collection')).not.toBeInTheDocument();
    });

    it('disables delete button when isDeleting is true', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} />);

      const deleteButton = screen.getByRole('button', { name: /deleting/i });
      expect(deleteButton).toBeDisabled();
    });

    it('enables delete button when isDeleting is false', () => {
      render(<DrawerActions {...defaultProps} isDeleting={false} />);

      const deleteButton = screen.getByRole('button', { name: /delete collection/i });
      expect(deleteButton).not.toBeDisabled();
    });

    it('defaults to not deleting when isDeleting prop not provided', () => {
      render(<DrawerActions {...defaultProps} />);

      expect(screen.getByText('Delete Collection')).toBeInTheDocument();
      const deleteButton = screen.getByRole('button', { name: /delete collection/i });
      expect(deleteButton).not.toBeDisabled();
    });
  });

  describe('Dynamic Source Button', () => {
    it('shows "Add Source" text when uploader is closed', () => {
      render(<DrawerActions {...defaultProps} showUploader={false} />);

      expect(screen.getByText('Add Source')).toBeInTheDocument();
      expect(screen.queryByText('Close Uploader')).not.toBeInTheDocument();
    });

    it('shows "Add Source" text when showUploader prop not provided', () => {
      render(<DrawerActions {...defaultProps} />);

      expect(screen.getByText('Add Source')).toBeInTheDocument();
      expect(screen.queryByText('Close Uploader')).not.toBeInTheDocument();
    });

    it('shows "Close Uploader" text when uploader is open', () => {
      render(<DrawerActions {...defaultProps} showUploader={true} />);

      expect(screen.getByText('Close Uploader')).toBeInTheDocument();
      expect(screen.queryByText('Add Source')).not.toBeInTheDocument();
    });

    it('source button is never disabled even when delete is in progress', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} showUploader={false} />);

      const addButton = screen.getByRole('button', { name: /add source/i });
      expect(addButton).not.toBeDisabled();
    });

    it('close uploader button is never disabled even when delete is in progress', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} showUploader={true} />);

      const closeButton = screen.getByRole('button', { name: /close uploader/i });
      expect(closeButton).not.toBeDisabled();
    });
  });

  describe('Dynamic Crawl Button', () => {
    it('shows "Crawl Web" text when crawler is closed', () => {
      render(<DrawerActions {...defaultProps} showCrawler={false} />);

      expect(screen.getByText('Crawl Web')).toBeInTheDocument();
      expect(screen.queryByText('Close Crawler')).not.toBeInTheDocument();
    });

    it('shows "Crawl Web" text when showCrawler prop not provided', () => {
      render(<DrawerActions {...defaultProps} />);

      expect(screen.getByText('Crawl Web')).toBeInTheDocument();
      expect(screen.queryByText('Close Crawler')).not.toBeInTheDocument();
    });

    it('shows "Close Crawler" text when crawler is open', () => {
      render(<DrawerActions {...defaultProps} showCrawler={true} />);

      expect(screen.getByText('Close Crawler')).toBeInTheDocument();
      expect(screen.queryByText('Crawl Web')).not.toBeInTheDocument();
    });
  });

  describe('Button Presence', () => {
    it('renders all three action buttons', () => {
      render(<DrawerActions {...defaultProps} />);

      expect(screen.getByRole('button', { name: /delete collection/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /add source/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /crawl web/i })).toBeInTheDocument();
    });
  });
});
